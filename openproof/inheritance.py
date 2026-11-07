"""Semantic RF classification-head expansion for incremental tasks."""

def expand_detector_state(old, fresh, old_names, new_names):
    import torch
    if new_names[:len(old_names)] != old_names or len(new_names) < len(old_names):
        raise RuntimeError('Incremental class order must preserve the complete old prefix')
    (previous, current) = (len(old_names), len(new_names))
    if old.keys() != fresh.keys():
        raise RuntimeError('RF architecture changed between tasks')
    (out, expanded) = ({}, [])
    for (key, value) in old.items():
        target = fresh[key]
        is_head = key.startswith('class_embed.') or '.enc_out_class_embed.' in key
        if is_head and value.shape[0] == previous + 1 and (target.shape[0] == current + 1):
            target = target.clone()
            target[:previous] = value[:previous]
            target[current] = value[previous]
            out[key] = target
            expanded.append(key)
        elif value.shape == target.shape:
            out[key] = value.clone()
        else:
            raise RuntimeError(f'Unapproved RF shape change: {key}: {value.shape} -> {target.shape}')
    if not expanded:
        raise RuntimeError('No RF classification head was expanded')
    return (out, expanded)

