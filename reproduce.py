#!/usr/bin/env python3
"""Reproduce the paper's exhibits and checks from archived or fresh simulations."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun', action='store_true',
                        help='rerun both original 20-seed simulation suites before building exhibits')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '1'
    env['MPLCONFIGDIR'] = str(root/'.runtime_cache/matplotlib')
    env['XDG_CACHE_HOME'] = str(root/'.runtime_cache')
    env['PYTHONDONTWRITEBYTECODE'] = '1'

    def run(script, *arguments):
        command = [sys.executable, str(root/script), *arguments]
        print('Running:', script, *arguments, flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)

    if args.rerun:
        run('simulation.py', '--seeds', '20')
        run('strategic.py', '--seeds', '20')
    run('make_exhibits.py')
    run('verify_results.py')
    print('Complete: tables and figures are in generated/.', flush=True)


if __name__ == '__main__':
    main()
