#!/usr/bin/env python3
"""Regenerate every table and figure from results/<ds>/minhash_results.csv.
Nothing is copied by hand.

Per dataset (results/<ds>/):
  fig_recall_vs_qps.png     recall@100 vs multi-thread end-to-end QPS (query sketching + rerank included)
  fig_recall_vs_memory.png  recall@100 vs total memory (MB): raw vectors included iff the config reranks
  tables.md                 frontiers, best config per recall target, max recall reached, sanity flags
Also results/summary.md (all datasets).

  python3 -P wmh/scripts/report.py [results_dir]
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reference categorical palette (light), first three slots: safe for all-pairs use in a scatter.
ALGO_COLOR = {'dart': '#2a78d6', 'ficws': '#eb6834', 'bmh2': '#1baf7a'}
ALGO_LABEL = {'dart': 'DartMinHash', 'ficws': 'FastICWS', 'bmh2': 'BagMinHash2'}
MODE_MARKER = {'A': 'o', 'B': '^'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
TARGETS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def pareto(df, x, y, y_better='high'):
    """Points not dominated: higher x (recall) and better y."""
    d = df.sort_values([x, y], ascending=[False, y_better != 'high'])
    keep, best = [], -np.inf if y_better == 'high' else np.inf
    for i, r in d.iterrows():
        v = r[y]
        if (y_better == 'high' and v > best) or (y_better == 'low' and v < best):
            keep.append(i)
            best = v
    return df.loc[keep].sort_values(x)


def config_label(r):
    s = f"{r.algorithm} {r['mode']} k={int(r.k)} b={int(r.slot_bits)}"
    if r['mode'] == 'B':
        s += f" L={int(r.L)} r={int(r.r)}"
    s += f" S={int(r.S)}" if r.rerank else ' no-rerank'
    return s


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(format(v, fmt[c]))
            elif isinstance(v, (float, np.floating)):
                cells.append(f'{v:.4g}')
            else:
                cells.append(str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def scatter_plot(ok, ex, ds, ycol, ylabel, y_better, path, title):
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for algo in [a for a in ALGO_COLOR if a in ok.algorithm.unique()]:
        for mode in ['A', 'B']:
            for rr in [0, 1]:
                d = ok[(ok.algorithm == algo) & (ok['mode'] == mode) & (ok.rerank == rr)]
                if d.empty:
                    continue
                c = ALGO_COLOR[algo]
                ax.scatter(d.recall_at_100, d[ycol], s=22, marker=MODE_MARKER[mode],
                           facecolors=c if rr else 'none', edgecolors=c, linewidths=1.0, alpha=0.85, zorder=3)
    fr = pareto(ok, 'recall_at_100', ycol, y_better)
    ax.plot(fr.recall_at_100, fr[ycol], color=INK, linewidth=2, zorder=4)
    ax.scatter(fr.recall_at_100, fr[ycol], s=26, color=INK, zorder=5)
    if not ex.empty:
        ax.scatter(ex.recall_at_100, ex[ycol], s=120, marker='*', color=INK, zorder=6)
        ax.annotate('exact (inverted index)', (ex.recall_at_100.iloc[0], ex[ycol].iloc[0]), textcoords='offset points',
                    xytext=(-8, 8), ha='right', color=INK2, fontsize=8)
    ax.set_yscale('log')
    ax.set_xlim(-0.02, 1.04)
    ax.set_xlabel('Recall@100 (exact weighted-Jaccard ground truth)', color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    # legend: algorithm (color), mode (shape), rerank (fill), frontier
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='s', linestyle='', color=ALGO_COLOR[a], label=ALGO_LABEL[a])
         for a in ALGO_COLOR if a in ok.algorithm.unique()]
    h += [Line2D([], [], marker='o', linestyle='', markerfacecolor='none', color=INK2, label='Mode A (scan)'),
          Line2D([], [], marker='^', linestyle='', markerfacecolor='none', color=INK2, label='Mode B (banded LSH)'),
          Line2D([], [], marker='o', linestyle='', color=INK2, label='filled = exact rerank'),
          Line2D([], [], color=INK, linewidth=2, label='Pareto frontier'),
          Line2D([], [], marker='*', linestyle='', color=INK, markersize=10, label='exact reference')]
    ax.legend(handles=h, fontsize=7.5, frameon=False, loc='best', labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return fr


def main(res_dir):
    summary = ['# Weighted-MinHash baseline: summary', '']
    for csv in sorted(glob.glob(os.path.join(res_dir, '*', 'minhash_results.csv'))):
        out = os.path.dirname(csv)
        ds = os.path.basename(out)
        df = pd.read_csv(csv)
        ex = df[df['mode'] == 'exact']
        mh = df[df['mode'].isin(['A', 'B'])]
        ok = mh[mh.status == 'ok'].copy()
        skipped = mh[mh.status != 'ok']
        lines = [f'# {ds}: weighted-MinHash baseline', '',
                 f'Rows: {len(ok)} measured configurations, {len(skipped)} skipped, {len(ex)} exact reference. '
                 f'N_db={int(df.n_db.iloc[0])}, queries={int(df.n_queries.iloc[0])}, threads (MT)={int(df.threads_mt.iloc[0])}, '
                 f'batch={int(ok.batch_size.iloc[0]) if len(ok) else "-"}, single-thread timing on the first '
                 f'{int(ok.n_queries_st.iloc[0]) if len(ok) else "-"} queries.', '',
                 'QPS = multi-thread end-to-end (query sketching and rerank included, 1 warm-up batch). '
                 'Memory = sketches + LSH tables + 8-byte id per vector (as faiss), plus the raw vectors (dense float32) only when the configuration reranks '
                 '(`mem_total_mb`).', '']

        # sanity flags
        bad_t = ok[ok.timing_check != 'ok']
        bad_d = ok[ok.st_mt_identical != 'yes']
        bad_g = ok[ok.diag_check != 'ok'] if 'diag_check' in ok else ok.iloc[0:0]
        bad_o = ok[(ok.recall_at_100 > ok.shortlist_recall_at_100 + 1e-12) |
                   (ok.shortlist_recall_at_100 > ok.candidate_recall_at_100 + 1e-12)] if 'diag_check' in ok else ok.iloc[0:0]
        slow = ok[ok.qps_mt < float(ex.qps_mt.iloc[0])] if len(ex) else ok.iloc[0:0]
        lines += ['## Sanity flags', '',
                  f'- timing_check != ok (end-to-end faster than filter-only): {len(bad_t)}',
                  f'- single-thread and multi-thread results differ: {len(bad_d)}',
                  f'- no-rerank shortlist recall != returned Recall@100: {len(bad_g)}',
                  f'- ordering violated (Recall@100 <= shortlist recall <= candidate recall): {len(bad_o)}',
                  f'- timed passes repeated because the machine slept: {int(ok.sleep_retries.sum()) if "sleep_retries" in ok else "n/a"}',
                  f'- configurations slower (MT QPS) than the exact inverted-index search: {len(slow)} of {len(ok)}'
                  + (f' (exact: {float(ex.qps_mt.iloc[0]):.1f} QPS MT, {float(ex.qps_st.iloc[0]):.1f} QPS ST)' if len(ex) else ''),
                  '']

        # max recall reached
        mx = ok.groupby(['algorithm', 'mode', 'rerank']).recall_at_100.max().reset_index()
        mx = mx.merge(ok, on=['algorithm', 'mode', 'rerank', 'recall_at_100']).sort_values('qps_mt', ascending=False) \
               .drop_duplicates(['algorithm', 'mode', 'rerank'])
        mx['config'] = mx.apply(config_label, axis=1)
        lines += ['## Highest Recall@100 reached', '',
                  md_table(mx.sort_values(['algorithm', 'mode', 'rerank']),
                           ['algorithm', 'mode', 'rerank', 'recall_at_100', 'recall_at_10', 'recall_at_1', 'qps_mt',
                            'mem_total_mb', 'config'],
                           {'recall_at_100': '.4f', 'recall_at_10': '.4f', 'recall_at_1': '.4f', 'qps_mt': '.1f',
                            'mem_total_mb': '.1f'}), '']

        # best per recall target
        rows = []
        for t in TARGETS:
            d = ok[ok.recall_at_100 >= t]
            if d.empty:
                rows.append({'target': t, 'best_qps_mt': np.nan, 'qps_config': 'not reached',
                             'min_mem_mb': np.nan, 'mem_config': 'not reached'})
                continue
            q = d.loc[d.qps_mt.idxmax()]
            m = d.loc[d.mem_total_mb.idxmin()]
            rows.append({'target': t, 'best_qps_mt': q.qps_mt, 'qps_config': config_label(q),
                         'min_mem_mb': m.mem_total_mb, 'mem_config': config_label(m)})
        lines += ['## Best configuration per Recall@100 target', '',
                  md_table(pd.DataFrame(rows), ['target', 'best_qps_mt', 'qps_config', 'min_mem_mb', 'mem_config'],
                           {'best_qps_mt': '.1f', 'min_mem_mb': '.1f'}), '']

        fq = scatter_plot(ok, ex, ds, 'qps_mt', 'Queries per second (multi-thread, end-to-end, log)', 'high',
                          os.path.join(out, 'fig_recall_vs_qps.png'), f'{ds}: recall vs QPS')
        fm = scatter_plot(ok, ex, ds, 'mem_total_mb', 'Total memory, MB (log; raw vectors counted iff rerank)', 'low',
                          os.path.join(out, 'fig_recall_vs_memory.png'), f'{ds}: recall vs memory')
        for name, fr in (('QPS', fq), ('memory', fm)):
            fr = fr.copy()
            fr['config'] = fr.apply(config_label, axis=1)
            lines += [f'## Pareto frontier: recall vs {name}', '',
                      md_table(fr, ['recall_at_100', 'shortlist_recall_at_100', 'candidate_recall_at_100', 'qps_mt', 'qps_st',
                              'mem_total_mb', 'candidates_per_query', 'config'],
                               {'recall_at_100': '.4f', 'shortlist_recall_at_100': '.4f', 'candidate_recall_at_100': '.4f',
                                'qps_mt': '.1f', 'qps_st': '.1f', 'mem_total_mb': '.1f',
                                'candidates_per_query': '.0f'}), '']
        if len(skipped):
            sk = skipped.copy()
            lines += ['## Skipped Mode B settings (decided on held-out queries only)', '',
                      md_table(sk, ['algorithm', 'k', 'L', 'r', 'status', 'heldout_candidates_est'],
                               {'heldout_candidates_est': '.0f'}), '']
        open(os.path.join(out, 'tables.md'), 'w').write('\n'.join(lines))
        summary += [f'## {ds}', ''] + lines[2:lines.index('## Best configuration per Recall@100 target') + 3] + ['']
        print(f'{ds}: wrote tables.md and 2 figures ({len(ok)} configurations)')
    open(os.path.join(res_dir, 'summary.md'), 'w').write('\n'.join(summary))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
