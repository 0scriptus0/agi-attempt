from dataclasses import dataclass

@dataclass(frozen=True)
class PermissionProfile:
    name: str
    workspace_only: bool
    allow_network: bool
    require_approval_for_destructive: bool

PROFILES = {
    'restricted': PermissionProfile('restricted', True, False, True),
    'workspace': PermissionProfile('workspace', True, True, True),
    'unrestricted': PermissionProfile('unrestricted', False, True, False),
}

def get_profile(name: str) -> PermissionProfile:
    if name not in PROFILES:
        raise ValueError(f'unknown permission profile: {name}')
    return PROFILES[name]
