#!/usr/bin/env python3
"""Summarize Step 0 estimator-accuracy output (step0_estimator_raw.csv) into
results/<dataset>/step0_summary.md.  All numbers come from the CSV.

  bias  = mean(est - J)           MAE = mean|est - J|        RMSE = sqrt(mean (est-J)^2)
  rmse/theory = RMSE / sqrt(mean J(1-J)/k)   (1.0 for an unbiased estimator with
                independent slots; >1 means extra error)
  bias_t = mean over seeds of the per-seed bias / (sd over seeds / sqrt(#seeds)).
           Pairs within one seed share the same hash functions, so seeds are the
           independent replicates.
"""
import sys, os
import numpy as np, pandas as pd

def agg(df, keys):
    g = df.groupby(keys, sort=True)
    s = g[['n', 'sum_J', 'sum_est', 'sum_err', 'sum_abs_err', 'sum_sq_err', 'sum_var_unit']].sum()
    out = pd.DataFrame({'n_pairs': s.n, 'mean_J': s.sum_J / s.n, 'mean_est': s.sum_est / s.n,
                        'bias': s.sum_err / s.n, 'MAE': s.sum_abs_err / s.n, 'RMSE': np.sqrt(s.sum_sq_err / s.n)})
    k = out.index.get_level_values('k') if 'k' in keys else None
    if k is not None:
        out['rmse/theory'] = out.RMSE / np.sqrt(s.sum_var_unit / s.n / np.asarray(k))
    # per-seed bias -> t statistic
    ps = df.groupby(keys + ['seed'])[['sum_err', 'n']].sum()
    ps = (ps.sum_err / ps.n).groupby(level=list(range(len(keys))))
    out['bias_sd_seeds'] = ps.std()
    out['bias_t'] = out.bias / (ps.std() / np.sqrt(ps.count()))
    out['n_seeds'] = ps.count()
    out['n_pairs'] = (out.n_pairs / df.groupby(keys).seed.nunique()).astype(int)  # pairs per seed (per group)
    return out.reset_index()

def md(df, floatfmt='.4f'):
    """Markdown table without the optional `tabulate` dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return 'nan' if np.isnan(v) else format(v, floatfmt)
        return str(v)
    rows = [list(df.columns)] + [[cell(v) for v in r] for r in df.itertuples(index=False)]
    out = ['| ' + ' | '.join(rows[0]) + ' |', '|' + '|'.join('---' for _ in rows[0]) + '|']
    out += ['| ' + ' | '.join(r) + ' |' for r in rows[1:]]
    return '\n'.join(out)

def load(res_dir, name):
    """Base file plus the seeded-BagMinHash2 file (<name>_bmh2.csv) when present;
    bmh2 rows in the base file come from an older unseeded build and are dropped."""
    df = pd.read_csv(os.path.join(res_dir, name + '.csv'))
    extra = os.path.join(res_dir, name + '_bmh2.csv')
    if os.path.exists(extra):
        df = pd.concat([df[df.algo != 'bmh2'], pd.read_csv(extra)], ignore_index=True)
    return df

def main(res_dir):
    df = load(res_dir, 'step0_estimator_raw')
    seeds = sorted(df.seed.unique())
    base = df[(df.scale == 1) & (df.bits == 64)]
    lines = [f'# Step 0 estimator accuracy: {os.path.basename(os.path.abspath(res_dir))}', '',
             f'Seeds: {", ".join(map(str, seeds))} ({len(seeds)} independent replicates of all hash functions). '
             'Pairs are drawn from the held-out "train" queries only. '
             'Pair types: nn = exact top-100 neighbours, strat = exact ranks 200..100000, random = uniform db rows.', '',
             'J_w buckets are [b/10, (b+1)/10). Estimates in tables 1-3 use full 64-bit ids (no truncation, no correction).', '']

    t1 = agg(base, ['algo', 'pair_type', 'k'])
    lines += ['## 1. Error by algorithm, pair type and k (64-bit ids)', '', md(t1), '']

    nn = base[base.pair_type.isin(['nn', 'strat'])]
    t2 = agg(nn[nn.k.isin([64, 256])], ['algo', 'k', 'bucket'])
    t2['bucket'] = t2.bucket.map(lambda b: f'[{b/10:.1f},{(b+1)/10:.1f})')
    lines += ['## 2. Error by J_w bucket (nn + strat pairs, 64-bit ids)', '', md(t2), '']

    tr = df[(df.scale == 1) & (df.k == 128)]
    t3 = agg(tr, ['algo', 'pair_type', 'bits', 'corrected'])
    lines += ['## 3. Slot truncation at k=128 (corrected=1 applies (p - 2^-b)/(1 - 2^-b))', '', md(t3), '']

    sc = df[(df.bits == 64) & df.k.isin([64, 256])]
    t4 = agg(sc, ['algo', 'k', 'pair_type', 'scale'])
    lines += ['## 4. Scaling all weights by a constant (J_w unchanged; 64-bit ids)', '', md(t4), '']

    bias_path = os.path.join(res_dir, 'step0_bias_raw.csv')
    if os.path.exists(bias_path):
        bd = load(res_dir, 'step0_bias_raw')
        nb = bd.seed.nunique()
        t5 = agg(bd[bd.bits == 64], ['algo', 'k', 'pair_type'])
        lines += [f'## 5. Bias test with {nb} fresh seeds (scale 1, 64-bit ids)', '',
                  'Seeds here are independent of tables 1-4. Under "no bias", bias_t is roughly t-distributed '
                  f'with {nb - 1} degrees of freedom (|t| > 2.7 has p < 0.01).', '', md(t5), '']
        dj = agg(bd[(bd.pair_type == 'disjoint') & (bd.corrected == 0)], ['algo', 'k', 'bits'])
        dj['false_match_rate'] = dj.mean_est
        dj['expected'] = np.where(dj.bits < 64, 2.0 ** -dj.bits.clip(upper=63), 0.0)
        lines += ['## 6. Disjoint-support pairs (J_w = 0): slot false-match rate, uncorrected', '',
                  md(dj[['algo', 'k', 'bits', 'n_pairs', 'false_match_rate', 'expected', 'bias_sd_seeds']], '.5f'), '']

    path = os.path.join(res_dir, 'step0_summary.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {path}')

if __name__ == '__main__':
    main(sys.argv[1])
