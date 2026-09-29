from dataclasses import dataclass
from typing import Callable

@dataclass
class Verification:
    passed: bool
    evidence: list[str]
    reason: str = ''

def verify(checks: list[tuple[str, Callable[[], bool]]]) -> Verification:
    evidence=[]
    for name, check in checks:
        try: ok=bool(check())
        except Exception as exc:
            evidence.append(f'{name}: exception: {exc!r}')
            return Verification(False,evidence,'verification check raised')
        evidence.append(f"{name}: {'passed' if ok else 'failed'}")
        if not ok: return Verification(False,evidence,f'check failed: {name}')
    return Verification(True,evidence,'all checks passed')
