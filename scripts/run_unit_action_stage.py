"""Sequential CUDA PPO comparisons and a fresh 20M run for one reference type.

This host orchestrator does no simulation. Child learners use run_gpu.sh; no
checkpoint is accepted, so every requested stage starts from random weights.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--root', default='results/unit-actions-20261009')
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument('--baseline', help='Prefix of paired 5M comparison runs')
    baseline.add_argument('--baseline-run', help='Previous stage fresh seed-42 20M run')
    parser.add_argument('--wandb-mode', choices=('online', 'offline'), default='offline')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output = (workspace / args.root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def read(name):
        return json.loads((output / name / 'results.json').read_text())

    def run(name, transitions, seed, mode):
        command = ['bash', 'scripts/run_gpu.sh', 'benchmark_number_grid.py',
                   '--total-timesteps', str(transitions), '--seed', str(seed),
                   '--output', str(output / name), '--wandb-mode', mode,
                   '--wandb-project', 'numbergrid', '--wandb-entity', 'sergey3784',
                   '--wandb-name', name+'-20261009']
        (output / (args.stage+'-status.json')).write_text(json.dumps(dict(phase=name)))
        with (output / (name+'.log')).open('w') as stdout, (output / (name+'.errors.log')).open('w') as stderr:
            result = subprocess.run(command, cwd=workspace, stdout=stdout, stderr=stderr)
        return result.returncode

    before, after = [], []
    for seed in ((42, 43) if args.baseline else ()):
        name = f'{args.stage}-bench-{seed}'
        if run(name, 5_000_000, seed, 'offline'):
            raise SystemExit('Comparison run failed: '+name)
        before.append(read(f'{args.baseline}-{seed}')['mean_steps_per_second'])
        after.append(read(name)['mean_steps_per_second'])
    comparison = None
    if before:
        regression = 1-sum(after)/sum(before)
        comparison = dict(before=before, after=after, regression_fraction=regression)
        (output / (args.stage+'-comparison.json')).write_text(json.dumps(comparison, indent=2))
        if regression >= .07:
            raise SystemExit('Repeated PPO regression >=7%; optimize and review before proceeding.')

    name = args.stage+'-20m'
    result = run(name, 20_000_000, 42, args.wandb_mode)
    if result and args.wandb_mode == 'online':
        error = (output / (name+'.errors.log')).read_text()
        # Only a confirmed service permission error before learner setup permits
        # an offline retry; training failures must never be silently retried.
        log = (output / (name+'.log')).read_text()
        if ('403' in error or 'PERMISSION_ERROR' in error) and '"phase": "initializing"' not in log:
            name += '-offline'
            result = run(name, 20_000_000, 42, 'offline')
    if result:
        raise SystemExit('20M training failed: '+name)
    trained = read(name)
    if (trained['training_steps'] != 20_000_000 or not trained['weights_changed']
            or not trained['checkpoint_roundtrip_verified'] or trained['backend'] != 'gpu'):
        raise SystemExit('20M checkpoint verification failed')
    if args.baseline_run:
        previous = read(args.baseline_run)
        for key in ('seed', 'training_steps', 'device', 'jax', 'num_envs', 'rollout_length'):
            if previous[key] != trained[key]:
                raise SystemExit('Non-comparable 20M baseline: '+key)
        regression = 1-trained['mean_steps_per_second']/previous['mean_steps_per_second']
        comparison = dict(before=[previous['mean_steps_per_second']],
                          after=[trained['mean_steps_per_second']], regression_fraction=regression)
        (output / (args.stage+'-comparison.json')).write_text(json.dumps(comparison, indent=2))
        if regression >= .07:
            raise SystemExit('PPO regression >=7%; repeat matched measurements and optimize before accepting.')
    (output / (args.stage+'-status.json')).write_text(json.dumps(dict(
        phase='complete', results=name, comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
