#!/usr/bin/env python3
"""Run the full pipeline on every dataset under <data_root> (no names hard-coded):
ground truth (if missing), the exact reference, then search runs for each
(algorithm, k). Each run writes results/<ds>/runs/<algo>_k<k>.csv; existing run
files are skipped, so the sweep can be resumed. Finally the run files are
concatenated into results/<ds>/minhash_results.csv.

  python3 -P wmh/scripts/sweep.py [--data data] [--results results] [--algos dart,ficws,bmh2]
"""
import argparse, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, 'wmh', 'bin')


def git_head(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def power_ok():
    """Timings are only meaningful on AC power with Low Power Mode off (macOS)."""
    try:
        batt = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
        settings = subprocess.check_output(['pmset', '-g'], text=True)
    except Exception:
        return True, 'pmset unavailable'
    lpm = any(l.split()[:2] == ['lowpowermode', '1'] for l in settings.splitlines())
    ac = "'AC Power'" in batt
    return ac and not lpm, f"AC={ac} lowpowermode={int(lpm)}"


def run(cmd, log):
    print('$ ' + ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        f.write('$ ' + ' '.join(cmd) + '\n')
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        sys.exit(f'command failed ({p.returncode}); see {log}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--algos', default='dart,ficws,bmh2')
    ap.add_argument('--ks', default='16,32,64,128,256,512,1024')
    ap.add_argument('--bmh2-max-k', type=int, default=256, help='BagMinHash2 (optional) is ~30x slower to sketch')
    ap.add_argument('--mem-budget-gb', type=float, default=400.0)
    ap.add_argument('--seed', default='20261014')
    ap.add_argument('--allow-battery', action='store_true', help='run even on battery / Low Power Mode')
    args = ap.parse_args()
    ok, why = power_ok()
    print(f'power: {why}', flush=True)
    if not ok and not args.allow_battery:
        sys.exit('refusing to time on battery or in Low Power Mode (use --allow-battery to override)')

    up = git_head(os.path.join(ROOT, 'third_party', 'dartminhash'))
    xx = git_head(os.path.join(ROOT, 'third_party', 'xxhash'))
    for ds_dir in sorted(glob.glob(os.path.join(args.data, '*', 'manifest.json'))):
        ds_dir = os.path.dirname(ds_dir)
        ds = os.path.basename(ds_dir)
        out = os.path.join(args.results, ds)
        runs = os.path.join(out, 'runs')
        os.makedirs(runs, exist_ok=True)
        log = os.path.join(out, 'sweep.log')
        n_db = json.load(open(os.path.join(ds_dir, 'manifest.json')))['stats']['db']['N']

        if not (os.path.exists(os.path.join(ds_dir, 'gt_query.bin')) and os.path.exists(os.path.join(ds_dir, 'gt_train.bin'))):
            run([os.path.join(BIN, 'groundtruth'), ds_dir], log)
        exact_csv = os.path.join(runs, 'exact.csv')
        if not os.path.exists(exact_csv):
            if os.path.exists(exact_csv + '.tmp'):
                os.remove(exact_csv + '.tmp')
            run([os.path.join(BIN, 'search'), '--data', ds_dir, '--out', exact_csv + '.tmp', '--exact',
                 '--upstream-git', up], log)
            os.replace(exact_csv + '.tmp', exact_csv)

        for algo in args.algos.split(','):
            for k in map(int, args.ks.split(',')):
                if algo == 'bmh2' and k > args.bmh2_max_k:
                    continue
                # 64-bit db sketches must fit in the memory budget together with everything else
                if n_db * k * 8 / 1e9 > 0.65 * args.mem_budget_gb:
                    print(f'{ds} {algo} k={k}: skipped, 64-bit sketches need {n_db * k * 8 / 1e9:.1f} GB', flush=True)
                    continue
                csv = os.path.join(runs, f'{algo}_k{k}.csv')
                if os.path.exists(csv):
                    continue
                if os.path.exists(csv + '.tmp'):
                    os.remove(csv + '.tmp')  # partial output of an interrupted run; search appends
                run([os.path.join(BIN, 'search'), '--data', ds_dir, '--out', csv + '.tmp', '--algo', algo, '--k', str(k),
                     '--seed', args.seed, '--mem-budget-gb', str(args.mem_budget_gb), '--upstream-git', up,
                     '--xxhash-git', xx], log)
                os.replace(csv + '.tmp', csv)  # only complete runs count as done

        files = sorted(glob.glob(os.path.join(runs, '*.csv')))
        with open(os.path.join(out, 'minhash_results.csv'), 'w') as f:
            for i, p in enumerate(files):
                # runs written before the CSV-quoting fix carried this unquoted comma-containing value
                lines = open(p).read().replace('yes (qps_st, qps_mt); no (qps_mt_filter_only)',
                                               'yes for qps_st and qps_mt; no for qps_mt_filter_only').splitlines()
                f.write('\n'.join(lines if i == 0 else lines[1:]) + '\n')
        print(f'{ds}: wrote {out}/minhash_results.csv from {len(files)} run files', flush=True)


if __name__ == '__main__':
    main()
