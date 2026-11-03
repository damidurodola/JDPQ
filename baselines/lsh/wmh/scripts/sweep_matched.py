#!/usr/bin/env python3
"""MinHash + banded-LSH (Mode B) sweep whose code sizes match the PQ sweep.

PQ stores M*nbits/8 bytes per vector (0.25 .. 160 B in the PQ grid). A MinHash
sketch stores k*b/8 bytes, so for every k only the slot widths b with
k*b/8 <= --max-code-bytes are run. Everything else (data, ground truth, rerank
sizes S, memory accounting) is shared with sweep.py; multi-thread only (--st-queries 0: no single-thread pass).

The only output is results/<ds>/minhash_results.csv. Every run is written to a temporary
directory and its rows are appended to that file once it has finished; the temporary files
are deleted. Re-running resumes: an (algorithm, k) whose rows are already in the CSV with
the same thread count is skipped (rows with another thread count are dropped and redone).
Per-config progress is printed to stdout only if --log is not given; --log FILE keeps it.

  python3 -P wmh/scripts/sweep_matched.py --datasets cemetery --threads 10
"""
import argparse, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, 'wmh', 'bin')


def git_head(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


import csv, shutil, tempfile  # noqa: E402

EXACT = 'exact_inverted_index'


def read_rows(path):
    if not os.path.exists(path):
        return None, []
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def run(cmd, log):
    print('$ ' + ' '.join(cmd), flush=True)
    out = open(log, 'a') if log else subprocess.DEVNULL
    try:
        if log:
            out.write('$ ' + ' '.join(cmd) + '\n')
            out.flush()
        p = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    finally:
        if log:
            out.close()
    if p.returncode != 0:
        sys.exit(f'command failed ({p.returncode})' + (f'; see {log}' if log else '; rerun with --log FILE'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--datasets', default='', help='comma list; default = every dataset under --data')
    ap.add_argument('--algos', default='dart,ficws,bmh2')
    ap.add_argument('--ks', default='1,2,4,8,16,32,64,128,256,512')
    ap.add_argument('--bits', default='1,2,4,8,16,64')
    ap.add_argument('--r', default='1,2,4,8,16,32,64')
    ap.add_argument('--S', default='0,100,500,1000,2000')
    ap.add_argument('--max-code-bytes', type=float, default=160.0, help='largest PQ code in the PQ grid (M=128, nbits=10)')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--mem-budget-gb', type=float, default=400.0)
    ap.add_argument('--prune-frac', type=float, default=0.5,
                    help='skip a Mode B setting if held-out queries get more than this fraction of the database as candidates '
                         '(use ~0.01 for 10M-vector databases)')
    ap.add_argument('--seed', default='20261014')
    ap.add_argument('--log', default='', help='optional file for the search programs output (default: discarded)')
    args = ap.parse_args()

    up = git_head(os.path.join(ROOT, 'third_party', 'dartminhash'))
    xx = git_head(os.path.join(ROOT, 'third_party', 'xxhash'))
    names = [n for n in args.datasets.split(',') if n] or sorted(
        os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(args.data, '*', 'manifest.json')))
    for ds in names:
        ds_dir = os.path.join(args.data, ds)
        out = os.path.join(args.results, ds)
        os.makedirs(out, exist_ok=True)
        out_csv = os.path.join(out, 'minhash_results.csv')
        fields, rows = read_rows(out_csv)
        keep = [r for r in rows if r['threads_mt'] == str(args.threads)]
        if len(keep) != len(rows):
            print(f'{ds}: dropping {len(rows) - len(keep)} rows measured with another thread count', flush=True)
            with open(out_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(keep)
            rows = keep
        done = {(r['algorithm'], r['k'] if r['algorithm'] != EXACT else '') for r in rows}

        def append(tmp_csv):
            nf, nr = read_rows(tmp_csv)
            with open(out_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=nf)
                if f.tell() == 0:
                    w.writeheader()
                w.writerows(nr)

        def search(extra):
            tmp = tempfile.mkdtemp(prefix='.tmp_', dir=out)
            try:
                tmp_csv = os.path.join(tmp, 'run.csv')
                run([os.path.join(BIN, 'search'), '--data', ds_dir, '--out', tmp_csv, '--threads', str(args.threads),
                     '--st-queries', '0', '--upstream-git', up] + extra, args.log)
                append(tmp_csv)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        if (EXACT, '') not in done:
            search(['--exact'])
        for algo in args.algos.split(','):
            for k in map(int, args.ks.split(',')):
                bits = [b for b in map(int, args.bits.split(',')) if k * b / 8 <= args.max_code_bytes]
                if not bits or (algo, str(k)) in done:
                    continue
                search(['--algo', algo, '--k', str(k), '--modes', 'B', '--bits', ','.join(map(str, bits)),
                        '--r', args.r, '--S', args.S, '--seed', args.seed, '--mem-budget-gb', str(args.mem_budget_gb), '--prune-frac', str(args.prune_frac),
                        '--xxhash-git', xx])
        _, rows = read_rows(out_csv)
        print(f'{ds}: {out_csv} has {len(rows)} rows', flush=True)


if __name__ == '__main__':
    main()
