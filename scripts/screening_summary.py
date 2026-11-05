"""Keep completed experiment identity separate from a resume/skip operation."""
from datetime import datetime
import json
import statistics


def screening_summary(variant, config, result, root, shared_digest, elapsed, train_times):
    status = result['status']
    if status not in ('completed', 'skipped'):
        raise ValueError('Only successful or verified skipped operations have a completed summary')
    artifact_attempt = result['verified_attempt'] if status == 'skipped' else result['attempt']
    median = statistics.median(train_times[4:]) if len(train_times) > 4 else None
    timing_source = 'measured_training_operation'
    training_elapsed = elapsed
    if status == 'skipped':
        folder = root / 'training' / result['run_id'] / 'attempts' / artifact_attempt
        original = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if (original['status'] != 'completed' or original['attempt'] != artifact_attempt
                or original['run_id'] != result['run_id']
                or original.get('initialization', {}).get('state_digest') != shared_digest):
            raise ValueError('Verified completed experiment identity mismatch')
        path = root / (variant + '-result.json')
        previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        if (previous is not None and previous.get('status') == 'completed'
                and previous.get('attempt') == artifact_attempt and previous.get('run_id') == result['run_id']
                and previous.get('shared_state_digest') == shared_digest
                and previous.get('optimizer_steps_budget') == config.budget['optimizer_steps']):
            training_elapsed = previous.get('elapsed_seconds')
            median = previous.get('median_train_step_seconds')
            timing_source = previous.get('elapsed_seconds_source', 'measured_training_operation')
        else:
            # A missing/legacy poisoned summary cannot recover per-step timing. Never use skip time.
            training_elapsed = (datetime.fromisoformat(original['finished_at']) -
                                datetime.fromisoformat(original['started_at'])).total_seconds()
            median = None
            timing_source = 'completed_manifest_wall_time'
    return {'variant': variant, 'status': 'completed', 'run_id': result['run_id'],
            'attempt': artifact_attempt, 'verified_attempt': artifact_attempt,
            'operation_status': status, 'operation_attempt': result['attempt'],
            'operation_elapsed_seconds': elapsed,
            'elapsed_seconds': training_elapsed, 'elapsed_seconds_source': timing_source,
            'shared_state_digest': shared_digest, 'optimizer_steps_budget': config.budget['optimizer_steps'],
            'median_train_step_seconds': median}
