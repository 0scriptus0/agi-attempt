#!/usr/bin/env python3
"""Lewi local agent runtime: trained model plus executable cognitive loop."""
from __future__ import annotations
import json, os, re, subprocess, tempfile, time, uuid
from datetime import date
from pathlib import Path
from typing import Any
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
LEARNING = ROOT / "learning_curve"
MEMORY_DIR = LEARNING / "memory"
INTERACTIONS_DIR = LEARNING / "interactions"
MEMORY_FILE = MEMORY_DIR / "memories.jsonl"
for p in (WORKSPACE, MEMORY_DIR, INTERACTIONS_DIR): p.mkdir(parents=True, exist_ok=True)
MODEL_PATH = os.environ.get("LEWI_MODEL", "checkpoints/lewi1-sft")
BASE_MODEL = os.environ.get("LEWI_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_AGENT_STEPS = int(os.environ.get("LEWI_MAX_AGENT_STEPS", "10"))
MAX_NEW_TOKENS = int(os.environ.get("LEWI_MAX_NEW_TOKENS", "700"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if DEVICE == "cuda" else torch.float32
FEATURE_FLAGS = {"code_exec": os.environ.get("LEWI_TOOL_CODE_EXEC", "1") == "1", "file_io": os.environ.get("LEWI_TOOL_FILE_IO", "1") == "1", "memory": os.environ.get("LEWI_TOOL_MEMORY", "1") == "1"}
app = FastAPI(title="Lewi Local Agent", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=[os.environ.get("LEWI_ALLOWED_ORIGIN", "http://localhost:8000")], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
MODEL = None
TOKENIZER = None
SYSTEM_PROMPT = '''You are Lewi 1, an agentic reasoning system. At each turn output EXACTLY one JSON object and no markdown:\n{"action":"reason","content":"..."}\n{"action":"memory_search","query":"..."}\n{"action":"memory_write","content":"...","memory_type":"lesson","scope":"global","applicable_when":"...","not_applicable_when":"...","evidence":"...","confidence":0.9}\n{"action":"code_exec","code":"..."}\n{"action":"file_read","path":"..."}\n{"action":"file_write","path":"...","content":"..."}\n{"action":"final","content":"..."}\nReason updates the plan. Memory is experience, not truth. Tools produce observations; inspect them and re-plan when they contradict expectations. Never claim success without evidence. After failure, change strategy instead of blindly repeating it. Only use final when the goal state is actually satisfied.'''


def load_model():
    global MODEL, TOKENIZER
    path = Path(MODEL_PATH)
    tokenizer_path = path if path.exists() and (path / "tokenizer_config.json").exists() else BASE_MODEL
    TOKENIZER = AutoTokenizer.from_pretrained(str(tokenizer_path))
    TOKENIZER.pad_token = TOKENIZER.pad_token or TOKENIZER.eos_token
    if path.exists() and (path / "adapter_config.json").exists():
        if PeftModel is None: raise RuntimeError("PEFT is required to load the trained adapter")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        MODEL = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
        MODEL = PeftModel.from_pretrained(MODEL, str(path))
    else:
        MODEL = AutoModelForCausalLM.from_pretrained(str(path if path.exists() else BASE_MODEL), torch_dtype=DTYPE, device_map="auto" if DEVICE == "cuda" else None)
    if DEVICE == "cpu": MODEL.to(DEVICE)
    MODEL.eval()


def safe_path(rel: str) -> Path:
    root = WORKSPACE.resolve(); path = (WORKSPACE / rel).resolve()
    if path != root and root not in path.parents: raise ValueError("path escapes workspace")
    return path


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower()) if t not in {"the","and","for","with","that","this","from","when","then","into","only","use","should","must","same"}}


def _memory_key(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item.get(k, "")).strip().lower() for k in ("memory_type","scope","content","applicable_when","not_applicable_when"))


def _load_memories() -> list[dict[str, Any]]:
    if not MEMORY_FILE.exists(): return []
    rows=[]
    for line in MEMORY_FILE.read_text(encoding="utf-8").splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    return rows


def _save_memories(rows: list[dict[str, Any]]) -> None:
    tmp = MEMORY_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(MEMORY_FILE)


def _relevance(query: str, item: dict[str, Any]) -> int:
    q = _terms(query)
    content = _terms(str(item.get("content", "")))
    applies = _terms(str(item.get("applicable_when", "")))
    excludes = _terms(str(item.get("not_applicable_when", "")))
    return len(q & content) * 5 + len(q & applies) * 2 - len(q & excludes) * 4


def memory_search(query: str, limit: int = 8) -> list[dict[str, Any]]:
    if not query.strip(): return []
    hits=[]
    for item in _load_memories():
        score = _relevance(query, item)
        if score >= 5: hits.append((score, float(item.get("confidence",0)), float(item.get("timestamp",0)), item))
    hits.sort(key=lambda x:(x[0],x[1],x[2]), reverse=True)
    out=[]; seen=set()
    for _,_,_,item in hits:
        key=_memory_key(item)
        if key in seen: continue
        seen.add(key); out.append(item)
        if len(out)>=limit: break
    return out


def memory_write(data: dict[str, Any]) -> dict[str, Any]:
    content = str(data.get("content", "")).strip()
    if not content: return {"ok": False, "error": "empty memory"}
    if not str(data.get("applicable_when", "")).strip() or not str(data.get("evidence", "")).strip(): return {"ok": False, "error": "memory requires applicable_when and evidence"}
    try: confidence = max(0.0, min(float(data.get("confidence", 0.7)), 1.0))
    except (TypeError, ValueError): confidence = 0.7
    item = {"id": str(uuid.uuid4()), "timestamp": time.time(), "memory_type": str(data.get("memory_type","lesson")), "scope": str(data.get("scope","global")), "content": content, "applicable_when": str(data.get("applicable_when","")), "not_applicable_when": str(data.get("not_applicable_when","")), "evidence": str(data.get("evidence","")), "confidence": confidence, "support_count": 1}
    rows = _load_memories(); key = _memory_key(item)
    for existing in rows:
        if _memory_key(existing) == key:
            existing["support_count"] = int(existing.get("support_count",1)) + 1
            existing["timestamp"] = item["timestamp"]
            existing["confidence"] = max(float(existing.get("confidence",0)), confidence)
            existing["evidence"] = item["evidence"]
            _save_memories(rows)
            return {"ok": True, "duplicate": True, "memory": existing}
    rows.append(item); _save_memories(rows)
    return {"ok": True, "memory": item}


def execute(action: dict[str, Any]) -> dict[str, Any]:
    name = action.get("action")
    if name == "memory_search": return {"ok": True, "results": memory_search(str(action.get("query","")))}
    if name == "memory_write": return memory_write(action)
    if name == "file_read" and FEATURE_FLAGS["file_io"]:
        path=safe_path(str(action.get("path","")))
        if not path.is_file(): return {"ok":False,"error":"file not found"}
        return {"ok":True,"path":str(path.relative_to(WORKSPACE)),"content":path.read_text(encoding="utf-8")[:20000]}
    if name == "file_write" and FEATURE_FLAGS["file_io"]:
        path=safe_path(str(action.get("path",""))); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(str(action.get("content","")),encoding="utf-8"); return {"ok":True,"path":str(path.relative_to(WORKSPACE))}
    if name == "code_exec" and FEATURE_FLAGS["code_exec"]:
        code=str(action.get("code",""))
        if not code.strip(): return {"ok":False,"error":"empty code"}
        script=None
        try:
            with tempfile.NamedTemporaryFile(mode="w",suffix=".py",dir=WORKSPACE,delete=False,encoding="utf-8") as f: f.write(code); script=Path(f.name)
            result=subprocess.run(["python3",str(script)],cwd=WORKSPACE,capture_output=True,text=True,timeout=30)
            return {"ok":result.returncode==0,"stdout":result.stdout[-5000:],"stderr":result.stderr[-5000:],"exit_code":result.returncode}
        except subprocess.TimeoutExpired: return {"ok":False,"error":"execution timed out after 30 seconds"}
        finally:
            if script: script.unlink(missing_ok=True)
    return {"ok":False,"error":f"unknown or disabled action: {name}"}


def extract_action(text: str) -> dict[str, Any] | None:
    for candidate in re.findall(r"\{.*?\}", text, flags=re.DOTALL)[::-1]:
        try:
            obj=json.loads(candidate)
            if isinstance(obj,dict) and obj.get("action"): return obj
        except json.JSONDecodeError: pass
    return None


def generate(transcript: list[dict[str,str]], memories: list[dict[str,Any]]) -> str:
    system=SYSTEM_PROMPT
    if memories:
        system += "\n\nRETRIEVED MEMORY:\n" + "\n".join(f"- {m.get('content')} | applies: {m.get('applicable_when')} | excludes: {m.get('not_applicable_when')} | confidence: {m.get('confidence')}" for m in memories)
    messages=[{"role":"system","content":system}]+transcript
    encoded=TOKENIZER.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt")
    input_ids=encoded["input_ids"] if hasattr(encoded,"keys") else encoded
    attention=encoded.get("attention_mask") if hasattr(encoded,"get") else None
    input_ids=input_ids.to(DEVICE)
    if attention is not None: attention=attention.to(DEVICE)
    with torch.no_grad(): output=MODEL.generate(input_ids=input_ids,attention_mask=attention,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=TOKENIZER.pad_token_id)
    return TOKENIZER.decode(output[0][input_ids.shape[1]:],skip_special_tokens=True).strip()


def run_agent(user_messages: list[dict[str,str]]) -> dict[str,Any]:
    trace=[]; transcript=list(user_messages)
    memories=memory_search(" ".join(m["content"] for m in user_messages if m["role"]=="user")) if FEATURE_FLAGS["memory"] else []
    allowed={"reason","memory_search","memory_write","code_exec","file_read","file_write","final"}
    for step in range(1,MAX_AGENT_STEPS+1):
        started=time.time(); raw=generate(transcript,memories); action=extract_action(raw); malformed=action is None
        if malformed: action={"action":"__invalid__","raw":raw}
        name=action.get("action"); event={"step":step,"action":action,"raw_model_output":raw}
        if name=="final":
            event["duration_ms"]=round((time.time()-started)*1000,2); trace.append(event)
            return {"reply":str(action.get("content","")),"trace":trace,"steps":step,"memories_used":memories,"stop_reason":"final"}
        if malformed: observation={"ok":False,"error":"model output was not valid JSON with an action field; output one exact JSON action and no markdown"}
        elif name not in allowed: observation={"ok":False,"error":f"unsupported action {name}; choose one of {sorted(allowed)}"}
        else:
            try: observation={"ok":True,"reasoning_update":str(action.get("content",""))} if name=="reason" else execute(action)
            except Exception as exc: observation={"ok":False,"error":str(exc)}
        observation["loop_state"]={"step":step,"last_action":name,"final_not_allowed_yet":True}
        event["observation"]=observation; event["duration_ms"]=round((time.time()-started)*1000,2); trace.append(event)
        transcript.append({"role":"assistant","content":raw}); transcript.append({"role":"user","content":"OBSERVATION: "+json.dumps(observation,ensure_ascii=False)})
        if name=="memory_search" and observation.get("ok"): memories=observation.get("results",[])
    return {"reply":"Agent stopped at the configured step limit without a final action.","trace":trace,"steps":MAX_AGENT_STEPS,"memories_used":memories,"stop_reason":"max_steps"}


def log_run(request: Any,result: dict[str,Any]):
    path=INTERACTIONS_DIR/f"{date.today().isoformat()}.jsonl"
    record={"id":str(uuid.uuid4()),"timestamp":time.time(),"model":MODEL_PATH,"request":request,**result}
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")

class ChatMessage(BaseModel):
    role:str=Field(pattern="^(user|assistant)$")
    content:str
class ChatRequest(BaseModel): messages:list[ChatMessage]

@app.on_event("startup")
def startup():
    if os.environ.get("LEWI_SKIP_MODEL_LOAD")!="1": load_model()

@app.get("/config")
def config(): return {"model":MODEL_PATH,"base_model":BASE_MODEL,"device":DEVICE,"max_agent_steps":MAX_AGENT_STEPS,"features":FEATURE_FLAGS}

@app.post("/chat")
def chat(req:ChatRequest):
    if not req.messages: raise HTTPException(400,"messages must be non-empty")
    if MODEL is None: raise HTTPException(503,"model is not loaded")
    messages=[m.model_dump() for m in req.messages]; result=run_agent(messages); log_run(messages,result); return result

@app.get("/memory")
def list_memory(): return {"memories":_load_memories()}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
