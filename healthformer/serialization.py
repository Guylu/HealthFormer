"""Restricted artifact loading and tensor/primitive checkpoint serialization."""
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

def to_safe(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, np.ndarray):
        return {'__healthformer_array__': True, 'dtype': str(value.dtype), 'values': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_safe(v) for v in value)
    if value is None or isinstance(value, (str, bool, int, float, torch.Tensor)):
        return value
    raise TypeError('Unsupported serialized value: ' + type(value).__name__)

def from_safe(value):
    if isinstance(value, dict):
        if value.get('__healthformer_array__') is True:
            dtype = np.dtype(value['dtype'])
            if dtype.hasobject:
                raise ValueError('Object arrays are not supported.')
            return np.array(value['values'], dtype=dtype)
        return {k: from_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(from_safe(v) for v in value)
    return value

def load_artifact(path, *, map_location='cpu'):
    return from_safe(torch.load(path, map_location=map_location, weights_only=True))

def artifact_path(settings):
    if settings.get('data_path'):
        return Path(settings['data_path'])
    raise ValueError('Provide data_path explicitly.')
