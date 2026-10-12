"""Generate a deterministic, entirely synthetic artifact for training smoke checks."""
import argparse
import json
from pathlib import Path
import torch


def create_fixture(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    names = ['rna_gene', 'microbiome_species', 'synthetic_DEXALoader', 'synthetic_BloodTestsLoader']
    ids = [100, 101, 119, 120, 121, 130, 140, 150]
    n, length, m = len(ids), 64, len(names)
    offsets = {i: i * 16 for i in range(m)}
    generator = torch.Generator().manual_seed(42)
    mods = torch.arange(length).remainder(m).repeat(n, 1)
    # One marker per year group; duplicate-marker cases are covered by regression checks.
    mods[mods == 2] = 3
    mods[:, [2, length // 2 + 2]] = 2
    relative = torch.randint(1, 14, (n, length), generator=generator)
    tokens = relative + mods * 16
    continuous = relative.float() + 0.5
    times = torch.ones(n, length, 7, dtype=torch.int32)
    times[:, :, 4] = 140
    times[:, length // 2:, 4] = 141
    times[:, :, 6] = 0
    artifact = dict(
        tokens=tokens, continuous_tokens=continuous, modalities=mods,
        preprocessing_provenance={'source': 'examples/make_training_fixture.py',
                                  'seed': 42, 'bin_fitting': 'fixed synthetic bins; no cohort fitting'},
        time_expanded=times, index=ids,
        cum_base_value_per_modality=offsets,
        modality_name_to_number={name: i for i, name in enumerate(names)},
        number_to_modality_name=dict(enumerate(names)),
        vals_per_modality={name: list(range(15)) for name in names},
        vocab_size=64, temporal_vocab_size=[8, 25, 61, 13, 151, 32, 3],
        time_dim_order=['day_of_week', 'hour', 'minute', 'month', 'year', 'day_of_month', 'sleep'],
        number_of_modalities=m, cat_cols=names[:2],
        bins={name: list(range(16)) for name in names[2:]},
    )
    torch.save(artifact, directory/'training.pt')
    settings = dict(
        data_path=str(directory/'training.pt'),
        checkpoint_dir=str(directory/'checkpoints'), accelerator='cpu', precision='32-true',
        devices=1, split_policy='modulo', percent_validation=20, epochs=1,
        n_embd=16, n_heads=1, n_layers=1, max_seq_length=length,
        reduce_context_in_train=False,
        batch_per_gpu=2, warmup_steps=1, evaluate=True, save=True,
        allow_wandb=False,
        Elis_augmentation_prob=0, elis_loss_weight=0,

        add_age_gender_as_pe=False, augmentation_chance=0,
        random_removal_chance=0, random_block_removal_chance=0,
        random_modality_subset_chance=0,
        random_modality_exclusion_chance=0, chance_to_remove_time_series_data=0,
    )
    (directory/'config.json').write_text(json.dumps(settings, indent=2)+'\n')
    return directory/'config.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/synthetic'))
    print(create_fixture(parser.parse_args().output))
