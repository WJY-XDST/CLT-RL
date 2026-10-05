"""Apply explicit replay overrides to config objects and actuator dictionaries."""


def apply_config_overrides(config, overrides):
    for key,value in overrides.items():
        node=config
        parts=key.split('.')
        for part in parts[:-1]:
            node=node[part] if isinstance(node,dict) else getattr(node,part)
        leaf=parts[-1]
        if isinstance(node,dict):
            if leaf not in node: raise ValueError(f'Unknown override: {key}')
            node[leaf]=value
        else:
            if not hasattr(node,leaf): raise ValueError(f'Unknown override: {key}')
            setattr(node,leaf,value)
