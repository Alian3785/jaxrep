"""Run the complete configured suite in two CUDA processes within ten minutes.

Usage: bash scripts/run_gpu.sh scripts/test_gpu.py --output results/gpu-tests/RUN
All fixtures and compilation are inside the timed region; no warm-up is hidden.
"""
import argparse
from contextlib import suppress
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-started-at',type=float,help='Parent launch epoch seconds, including WSL startup')
    args = parser.parse_args()
    if args.wall_started_at is not None and (not math.isfinite(args.wall_started_at) or args.wall_started_at > time.time()+1):
        parser.error('--wall-started-at must be a finite timestamp no later than this launch')
    if os.environ.get('JAX_PLATFORMS') != 'cuda':
        raise SystemExit('Run through scripts/run_gpu.sh; CUDA is mandatory.')
    root = Path(__file__).resolve().parents[1]
    paths = tomllib.loads((root/'pyproject.toml').read_text())['tool']['pytest']['ini_options']['testpaths']
    actions = 'stoix/tests/number_grid_reference_actions_test.py'
    if actions not in paths or len(paths) != len(set(paths)):
        raise SystemExit('Unexpected testpaths; refuse an incomplete/duplicate suite.')
    # Construction has independent faction graphs. Balance their compilation
    # against reference actions; keeping shared current-map tests together avoids
    # repeating their cached graph and leaves headroom under the wall-time limit.
    first = [actions,'stoix/tests/number_grid_buildings_test.py']
    if any(p not in paths for p in first):
        raise SystemExit('Missing configured construction/reference tests.')
    # Reuse the environment group's already compiled auto-reset graph.
    # Shared wrapper node IDs are excluded from actions, so every test runs once.
    shared = ('test_training_autoreset_retains_final_build_in_terminal_observation',)
    # The custom healer graph otherwise leaves the environment worker trailing.
    # Keep exactly one execution of it, after the actions group's shared graphs.
    healer = 'test_player_healing_mask_step_and_protections_agree'
    groups = {
        'actions':first+['stoix/tests/number_grid_variety_test.py::'+healer,
                         '-k',' and '.join('not '+name for name in shared)],
        'environment':[p for p in paths if p not in first]
                      +[first[1]+'::'+name for name in shared]+['-k','not '+healer],
    }
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    # Fresh per-run cache: both CUDA workers can reuse each other's compiled
    # graphs, while all compilation stays inside this measured cold run.
    cache = output/'compilation-cache'
    cache.mkdir()
    worker_env = {**os.environ,'JAX_ENABLE_COMPILATION_CACHE':'true',
                  'JAX_COMPILATION_CACHE_DIR':str(cache),
                  'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS':'1'}
    workers = {}
    streams = []
    results = {}
    # A parent timestamp measures the actual end-to-end command. Keep six
    # seconds for termination/reporting. Native WSL invocations use the same
    # budget; Windows launchers can include their startup with the timestamp.
    execution_limit = (594. if args.wall_started_at is None else
                       max(0.,594.-(time.time()-args.wall_started_at)))
    deadline = started+execution_limit
    try:
        for name,files in groups.items():
            stream = (output/(name+'.log')).open('w')
            streams.append(stream)
            command = [sys.executable,'-m','pytest','-v','--durations=20',*files]
            worker = subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=worker_env)
            workers[name] = worker
            print(json.dumps(dict(phase='started',group=name,pid=worker.pid,testpaths=files)),flush=True)
        while len(results) < len(workers):
            for name,worker in workers.items():
                code = worker.poll()
                if code is not None and name not in results:
                    results[name] = dict(exit_code=code,seconds=time.monotonic()-started,log=str(output/(name+'.log')))
                    print(json.dumps(dict(phase='finished',group=name,**results[name])),flush=True)
            if len(results) == len(workers):
                break
            if time.monotonic() >= deadline:
                print(f'Full CUDA suite exceeded its {execution_limit:.1f}s execution budget.',flush=True)
                break
            time.sleep(.25)
    finally:
        # Terminate only the process groups created above, never other GPU work.
        for worker in workers.values():
            if worker.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid,signal.SIGTERM)
        for worker in workers.values():
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid,signal.SIGKILL)
                worker.wait()
        for stream in streams:
            stream.close()
        summary = dict(seconds=time.monotonic()-started,execution_limit=execution_limit,wall_limit=600,
                       parent_started_at=args.wall_started_at,backend='cuda',groups=results,
                       compilation_cache=str(cache),cold_cache=True,
                       passed=len(results)==len(groups) and all(r['exit_code']==0 for r in results.values()))
        (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
