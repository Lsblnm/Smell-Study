#!/usr/bin/env python3
"""合并全部静态度量：组间比较 + 与结局差值的相关性。"""
import json, collections
import numpy as np
from scipy import stats

TAIL = ['babel__babel-14532', 'briannesbitt__carbon-2665', 'fluent__fluentd-3466',
        'jqlang__jq-2235', 'nlohmann__json-4237', 'php-cs-fixer__php-cs-fixer-7635']

s1 = {r['instance_id']: r for r in json.load(open('/tmp/fe_tail/static_metrics.json'))}
s2 = {r['instance_id']: r for r in json.load(open('/tmp/fe_tail/static_metrics2.json'))}
traj = json.load(open('/tmp/fe_tail/traj_metrics.json'))

data = {}
for iid, r in s1.items():
    d = dict(r); d.update(s2.get(iid, {}))
    a, b = traj[iid]['1|none'], traj[iid]['1|smell']
    d['d_recon'] = b['recon'] - a['recon']
    d['d_tools'] = b['tools'] - a['tools']
    d['d_time'] = (b['elapsed'] or 0) - (a['elapsed'] or 0)
    d['d_tok'] = b['tok'] - a['tok']
    data[iid] = d

OUT = ['d_recon', 'd_tools', 'd_time', 'd_tok']
skip = {'instance_id', 'lang'} | set(OUT)
metrics = [k for k in next(iter(data.values())) if k not in skip]


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def bh(rows, key='p'):
    rows.sort(key=lambda r: r[key])
    n = len(rows); prev = 1.0
    for i in range(n - 1, -1, -1):
        rows[i]['q'] = prev = min(prev, rows[i][key] * n / (i + 1))
    return rows


# ---------- 1. 组间比较 ----------
res = []
for m in metrics:
    a = [data[i][m] for i in TAIL if data.get(i, {}).get(m) is not None]
    b = [d[m] for k, d in data.items() if k not in TAIL and d.get(m) is not None]
    if len(a) < 5 or len(b) < 30:
        continue
    a, b = list(map(float, a)), list(map(float, b))
    if len(set(a + b)) < 3:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    res.append(dict(metric=m, mean_t=np.mean(a), mean_r=np.mean(b),
                    med_t=np.median(a), med_r=np.median(b), p=p, delta=cliffs(a, b)))
bh(res)

print('=== 右长尾(n=6) vs 其余(n=68)：Mann-Whitney ===')
print(f'{"metric":24s}{"尾均值":>11s}{"其余均值":>11s}{"尾中位":>9s}{"其余中位":>9s}'
      f'{"中位比":>8s}{"Cliffd":>8s}{"p":>9s}{"q":>7s}')
print('-' * 96)
for r in res:
    rt = r['med_t'] / r['med_r'] if r['med_r'] else float('inf')
    star = ' ***' if r['q'] < .05 else (' *' if r['p'] < .05 else '')
    print(f'{r["metric"]:24s}{r["mean_t"]:11.1f}{r["mean_r"]:11.1f}{r["med_t"]:9.1f}'
          f'{r["med_r"]:9.1f}{rt:8.2f}{r["delta"]:8.2f}{r["p"]:9.4f}{r["q"]:7.2f}{star}')

# ---------- 2. 相关性 ----------
print('\n=== 静态度量 vs 配对差值：Spearman (n=74) ===')
for out in OUT:
    rows = []
    for m in metrics:
        xs, ys = [], []
        for iid, d in data.items():
            if d.get(m) is None:
                continue
            xs.append(float(d[m])); ys.append(float(d[out]))
        if len(set(xs)) < 5:
            continue
        rho, p = stats.spearmanr(xs, ys)
        rows.append(dict(metric=m, rho=rho, p=p))
    bh(rows)
    top = [r for r in rows if r['p'] < 0.10][:8]
    print(f'\n-- {out} --')
    if not top:
        print('   无 p<0.10 的度量')
    for r in top:
        star = ' ***' if r['q'] < .05 else ' *'
        print(f'   {r["metric"]:24s} rho={r["rho"]:+.3f}  p={r["p"]:.4f}  q={r["q"]:.3f}{star}')

# ---------- 3. 语言分布 ----------
print('\n=== 语言分布 ===')
ct = collections.Counter(data[i]['lang'] for i in TAIL)
cr = collections.Counter(d['lang'] for k, d in data.items() if k not in TAIL)
allang = sorted(set(ct) | set(cr))
print(f'{"lang":10s}{"尾":>5s}{"其余":>6s}')
for l in allang:
    print(f'{l:10s}{ct.get(l,0):5d}{cr.get(l,0):6d}')
tj = sum(1 for i in TAIL if data[i]['lang'] == 'Java')
rj = sum(1 for k, d in data.items() if k not in TAIL and d['lang'] == 'Java')
tab = [[tj, 6 - tj], [rj, 68 - rj]]
print('Java vs 非Java 的 Fisher 精确检验 p =', stats.fisher_exact(tab)[1], tab)

json.dump({k: v for k, v in data.items()}, open('/tmp/fe_tail/merged.json', 'w'), indent=1, default=float)
