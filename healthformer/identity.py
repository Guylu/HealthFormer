"""Content identity for exact experiment continuation."""
import hashlib
from pathlib import Path
from omegaconf import OmegaConf
from .serialization import artifact_path
from .config import normalize_config

# These settings control locations or reporting, not the trained objective.
OPERATIONAL_SETTINGS = {
    'data_path', 'checkpoint_dir', 'resume_from', 'initialize_from',
    'allow_wandb', 'wandb_entity', 'wandb_project', 'save',
    'stop_early_now', 'log_wandb_metric_every_k',
}

def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def experiment_identity(config, split_manifest):
    settings = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    settings = normalize_config(settings)
    return {
        'schema_version': 9,
        'settings': {**{k: v for k, v in settings.items() if k not in OPERATIONAL_SETTINGS},
                     'optimizer': 'adamw', 'visit_definition': 'distinct_year'},
        'artifacts': {
            'training': file_digest(artifact_path(config)),
        },
        'split': split_manifest,
    }

def validate_identity(saved, current):
    if not isinstance(saved, dict) or saved.get('schema_version') != 9:
        raise ValueError('Checkpoint lacks a supported experiment identity; use initialize_from for weights only.')
    differences = []
    for group in ('settings', 'artifacts', 'split'):
        a, b = saved.get(group, {}), current.get(group, {})
        if group == 'settings':
            a, b = normalize_config(a), normalize_config(b)
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key) or (key in a) != (key in b):
                differences.append(f'{group}.{key}')
    if differences:
        raise ValueError('Exact resume identity differs: ' + ', '.join(differences))
