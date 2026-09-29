from __future__ import annotations
import json,time,uuid
from dataclasses import asdict,dataclass,field
from pathlib import Path
from threading import Event,Lock
from typing import Any,Callable

TERMINAL={'completed','failed','cancelled'}

@dataclass
class Task:
    id:str
    prompt:str
    status:str='queued'
    priority:int=0
    parent_id:str|None=None
    dependencies:list[str]=field(default_factory=list)
    attempts:int=0
    step:int=0
    created_at:float=field(default_factory=time.time)
    updated_at:float=field(default_factory=time.time)
    checkpoint:dict[str,Any]=field(default_factory=dict)
    result:dict[str,Any]|None=None
    error:str|None=None

class TaskStore:
    def __init__(self,root='.lewi'):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.events=self.root/'events.jsonl'; self.tasks=self.root/'tasks.jsonl'; self._lock=Lock(); self._state={}; self._load()
    def _load(self):
        if not self.events.exists(): return
        for line in self.events.read_text(encoding='utf-8').splitlines():
            if line.strip():
                e=json.loads(line); t=e.get('task')
                if t: self._state[t['id']]=Task(**t)
    def _append(self,task,event):
        task.updated_at=time.time(); payload={'event':event,'timestamp':task.updated_at,'task':asdict(task)}
        with self.events.open('a',encoding='utf-8') as f: f.write(json.dumps(payload,ensure_ascii=False)+'\n')
        with self.tasks.open('a',encoding='utf-8') as f: f.write(json.dumps(asdict(task),ensure_ascii=False)+'\n')
    def create(self,prompt,**kwargs):
        t=Task(id=str(uuid.uuid4()),prompt=prompt,**kwargs)
        with self._lock: self._state[t.id]=t; self._append(t,'created')
        return t
    def get(self,task_id): return self._state.get(task_id)
    def all(self): return list(self._state.values())
    def runnable(self):
        out=[]
        for t in self._state.values():
            if t.status not in {'queued','waiting_retry'}: continue
            if all(self.get(d) and self.get(d).status=='completed' for d in t.dependencies): out.append(t)
        return sorted(out,key=lambda t:(-t.priority,t.created_at))
    def checkpoint(self,task,event='checkpoint'):
        with self._lock: self._state[task.id]=task; self._append(task,event)

class Worker:
    def __init__(self,store:TaskStore,executor:Callable[[Task],str],poll_seconds=1.0):
        self.store=store; self.executor=executor; self.poll_seconds=poll_seconds; self.stop_event=Event()
    def stop(self): self.stop_event.set()
    def run_forever(self):
        while not self.stop_event.is_set():
            runnable=self.store.runnable()
            if not runnable: self.stop_event.wait(self.poll_seconds); continue
            task=runnable[0]; task.status='running'; task.attempts+=1; self.store.checkpoint(task,'started')
            try:
                status=self.executor(task)
                if status not in TERMINAL: raise RuntimeError(f'executor returned non-terminal status: {status}')
                task.status=status; self.store.checkpoint(task,'terminal')
            except Exception as exc:
                task.error=repr(exc); task.status='waiting_retry'; self.store.checkpoint(task,'executor_error')
                self.stop_event.wait(min(30.0,2.0**min(task.attempts,4)))
