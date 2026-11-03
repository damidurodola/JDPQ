#!/usr/bin/env python3
"""Collect data/*/manifest.json into results/dataset_stats.{csv,md}."""
import glob, json, os, sys
import pandas as pd

def main(data_root, out_dir):
    rows = []
    for m in sorted(glob.glob(os.path.join(data_root, '*', 'manifest.json'))):
        man = json.load(open(m))
        for split in ('db', 'query', 'train'):
            rows.append({'dataset': man['name'], 'split': split, **man['stats'][split]})
        rows[-3].update({f'cleaning.{k}': v for k, v in man['cleaning'].items()})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'dataset_stats.csv'), index=False)
    cols = ['dataset', 'split', 'N', 'd', 'nnz_mean', 'nnz_median', 'nnz_max', 'l1_min', 'l1_median', 'l1_max']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df[cols].itertuples(index=False):
        lines.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in r) + ' |')
    clean = [c for c in df.columns if c.startswith('cleaning.')]
    lines += ['', '| dataset | ' + ' | '.join(c[9:] for c in clean) + ' |', '|' + '---|' * (len(clean) + 1)]
    for r in df[df.split == 'db'].itertuples(index=False):
        d = r._asdict()
        lines.append(f"| {d['dataset']} | " + ' | '.join(str(int(df.loc[(df.dataset == d['dataset']) & (df.split == 'db'), c].iloc[0])) for c in clean) + ' |')
    open(os.path.join(out_dir, 'dataset_stats.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data', sys.argv[2] if len(sys.argv) > 2 else 'results')
