"""Registered CPU engineering runner for strict G2 source training and selection."""
import argparse
from pathlib import Path

import numpy as np
import torch

from loadData.spatial_protocol import forward_partition
from loadData.spatial_tiles import tile_slice
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.task_losses import detached_loss_values, probability_loss_components
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.g2_contract import fields, read_pinned, validate_contract, validate_lineage
from transfer.g2_observations import source_observations
from transfer.g2_source import build_source_model, require_shared_heads, source_labels
from transfer.g2_validation import evaluate_source_validation
from transfer.g3_training import _optimizer, head_hash, state_hash


SELECTION = 'first_min_source_macro_fused_nll'


def _source_inputs(catalog, contract, scene, entry):
    fields(entry, 'plan plan_sha256 train_observations train_labels val_observations val_labels superpixel_scale',
           'source runner entry')
    scale = entry['superpixel_scale']
    if type(scale) is not int or scale <= 0:
        raise ValueError('Source runner requires an explicit positive graph scale')
    plan = read_pinned(Path(entry['plan']), entry['plan_sha256'])
    names = ('train_observations', 'train_labels', 'val_observations', 'val_labels')
    payloads = {name: Path(entry[name]).read_bytes() for name in names}
    train = source_observations(catalog, contract, scene, 'train', plan, payloads['train_observations'])
    labels, counts = source_labels(catalog, contract, scene, 'source_train', plan, payloads['train_labels'])
    # Validate every registered validation input before spending the first update.
    source_observations(catalog, contract, scene, 'val', plan, payloads['train_observations'],
                        val_payload=payloads['val_observations'])
    source_labels(catalog, contract, scene, 'source_select', plan, payloads['val_labels'])
    tiles = [tile['id'] for tile in plan['tiles']
             if tile['partition'] == 'train' and (labels[tile_slice(tile)] > 0).any()]
    validation = tuple(payloads[name] for name in ('train_observations', 'val_observations', 'val_labels'))
    return dict(plan=plan, scale=scale, train=train, labels=labels, counts=counts, tiles=tiles,
                validation=validation)


def _update(model, head_key, source, tile_id, config, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observation = source['train'].tile(tile_id)
    outputs = forward_partition(model, head_key, observation, source['train'].wavelengths_nm.copy(),
                                source['scale'], wavelength_unit='nm')
    mapped = source['labels'].ravel()[observation.global_ids.ravel()]
    labels = torch.as_tensor(mapped, device=outputs[0].device)
    classes = len(source['counts'])
    mask = (labels > 0)[:, None].expand(-1, classes)
    onehot = torch.nn.functional.one_hot((labels - 1).clamp_min(0), classes).float() * mask
    loss = config.research.loss
    components = probability_loss_components(
        outputs, onehot, mask.float(), class_balance=loss.balanced,
        cnn_loss_weight=loss.cnn_loss_weight, graph_branch_loss_weight=loss.graph_branch_loss_weight,
        graph_loss_weights={'compactness': loss.graph_compactness_weight,
                            'entropy': loss.graph_entropy_weight, 'balance': loss.graph_balance_weight})
    if not torch.isfinite(components['total']):
        raise ValueError('Nonfinite G2 source training loss')
    components['total'].backward()
    optimizer.step()
    return detached_loss_values(components)


def _validation(catalog, contract, model, sources):
    model.eval()
    return [evaluate_source_validation(catalog, contract, model, scene, row['plan'], *row['validation'],
                                       superpixel_scale=row['scale']) for scene, row in sources.items()]


def run_source(registration_path, registration_sha256, expected_commit, out):
    code = code_snapshot()
    if code['commit'] != expected_commit:
        raise ValueError('Unexpected G2 runner commit')
    require_frozen_code(code)
    registration = read_pinned(Path(registration_path), registration_sha256)
    fields(registration, 'policy catalog catalog_sha256 contract contract_sha256 sources base model_seed '
                         'sampler_seed steps_per_source val_every_rounds selection', 'G2 source registration')
    if (registration['policy'] != 'g2_source_cpu_engineering_v1'
            or registration['selection'] != SELECTION
            or any(type(registration[k]) is not int or registration[k] <= 0 for k in ('steps_per_source', 'val_every_rounds'))
            or any(type(registration[k]) is not int or not 0 <= registration[k] < 2**32
                   for k in ('model_seed', 'sampler_seed'))):
        raise ValueError('Unsupported G2 source runner policy, seeds or budget')

    catalog = read_pinned(Path(registration['catalog']), registration['catalog_sha256'])
    contract = read_pinned(Path(registration['contract']), registration['contract_sha256'])
    if contract['catalog_sha256'] != registration['catalog_sha256']:
        raise ValueError('Contract catalog identity differs')
    validate_contract(catalog, contract)
    if not isinstance(registration['sources'], dict) or set(registration['sources']) != set(contract['sources']):
        raise ValueError('Source runner inputs must match registered sources exactly')
    out = external_directory(out)
    if out.exists():
        raise FileExistsError(out)

    model, config = build_source_model(catalog, contract, registration['base'], seed=registration['model_seed'])
    if model.backbone.coverage_mode != 'none':
        raise ValueError('G2 source releases do not carry coverage metadata')
    head_key = require_shared_heads(catalog, contract, model)
    sources = {name: _source_inputs(catalog, contract, name, registration['sources'][name])
               for name in contract['sources']}
    names = list(sources)
    steps = len(names) * registration['steps_per_source']
    model.training_config['g2_run'] = dict(registration_sha256=registration_sha256,
                                           contract_sha256=registration['contract_sha256'], selection=SELECTION)
    initial_state = state_hash(model)
    optimizer = _optimizer(model, config)
    rng = np.random.default_rng(registration['sampler_seed'])
    out.mkdir(parents=True, exist_ok=False)
    trace, history = [], []
    best = None

    for step in range(1, steps + 1):
        scene = names[(step - 1) % len(names)]
        row = sources[scene]
        tile_id = int(rng.choice(row['tiles']))
        loss = _update(model, head_key, row, tile_id, config, optimizer)
        trace.append(dict(step=step, scene=scene, tile_id=tile_id, loss=loss))
        if step % (len(names) * registration['val_every_rounds']) and step != steps:
            continue
        scores = _validation(catalog, contract, model, sources)
        macro_nll = float(np.mean([item['metrics']['fused']['nll'] for item in scores]))
        event = dict(step=step, macro_fused_nll=macro_nll, scenes=scores)
        if best is None or macro_nll < best['macro_fused_nll']:
            folder = out / f'step-{step:06d}'
            folder.mkdir()
            model.training_config['g2_run']['selected_step'] = step
            checkpoint = folder / 'model.pt'
            save_model_checkpoint(model, checkpoint)
            best = event
        history.append(event)

    selected = load_model_checkpoint(checkpoint, 'cpu')
    require_shared_heads(catalog, contract, selected)
    if _validation(catalog, contract, selected, sources) != best['scenes']:
        raise ValueError('Selected CPU checkpoint validation changed after strict reload')
    checksum = hash_file(checkpoint)
    access = []
    for scene in names:
        release = contract['source_releases'][scene]['train']
        access.extend([
            dict(scene=scene, phase='train_graph', partition='train',
                 observation_sha256=release['observation_sha256'], labels_sha256=None),
            dict(scene=scene, phase='source_train', partition='train', **release),
        ])
    for row in best['scenes']:
        access.extend(row['access'])
    lineage = dict(policy='g2_source_lineage_v1', contract_sha256=registration['contract_sha256'],
                   selected=checksum, records={checksum: dict(parents=[], access=access,
                       head=dict(kind='shared_semantic', ordered_classes=contract['semantic_classes'],
                                 state_sha256=head_hash(selected)))})
    validate_lineage(contract, lineage, registration['contract_sha256'])
    require_frozen_code(code)
    write_json(out / 'lineage.json', lineage)
    result = dict(status='completed_source_cpu_engineering', commit=expected_commit,
                  registration_sha256=registration_sha256, selection=SELECTION, selected_step=best['step'],
                  optimizer_updates=len(trace), updates_per_source={name: registration['steps_per_source'] for name in names},
                  initial_state_sha256=initial_state, selected_state_sha256=state_hash(selected),
                  selected_checkpoint=str(checkpoint.relative_to(out)), checkpoint_sha256=checksum,
                  sidecar_sha256=hash_file(checkpoint.with_suffix('.pt.json')),
                  lineage_sha256=hash_file(out / 'lineage.json'), history=history, trace=trace,
                  target_payload_access=False, query_accessed=False)
    write_json(out / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_source(args.registration, args.registration_sha256, args.expected_commit, args.out)


if __name__ == '__main__':
    main()
