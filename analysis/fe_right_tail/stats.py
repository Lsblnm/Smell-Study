#!/usr/bin/env python3
"""右长尾 vs 其余实例：传统静态度量的组间比较。"""
import json, math
import numpy as np
from scipy import stats

TAIL = ['babel__babel-14532', 'briannesbitt__carbon-2665', 'fluent__fluentd-3466',
        'jqlang__jq-2235', 'nlohmann__json-4237', 'php-cs-fixer__php-cs-fixer-7635']

rows = json.load(open('/tmp/fe_tail/static_metrics.json'))
by = {r['instance_id']: r for r in rows}
metrics = [k for k in rows[0] if k != 'instance_id']


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


res = []
for m in metrics:
    a = [by[i][m] for i in TAIL if by.get(i, {}).get(m) is not None]
    b = [r[m] for r in rows if r['instance_id'] not in TAIL and r.get(m) is not None]
    if len(a) < 4 or len(b) < 10:
        continue
    a = list(map(float, a)); b = list(map(float, b))
    try:
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        continue
    d = cliffs_delta(a, b)
    res.append(dict(metric=m, n_tail=len(a), n_rest=len(b),
                    mean_tail=np.mean(a), mean_rest=np.mean(b),
                    med_tail=np.median(a), med_rest=np.median(b),
                    p=p, delta=d))

# BH 校正
res.sort(key=lambda r: r['p'])
n = len(res)
prev = 1.0
for i in range(n - 1, -1, -1):
    q = min(prev, res[i]['p'] * n / (i + 1))
    res[i]['q'] = q
    prev = q

print(f'{"metric":26s} {"tail均值":>12s} {"其余均值":>12s} {"tail中位":>10s} {"其余中位":>10s} '
      f'{"倍数":>7s} {"Cliff d":>8s} {"p":>9s} {"q(BH)":>8s}')
print('-' * 118)
for r in res:
    ratio = r['mean_tail'] / r['mean_rest'] if r['mean_rest'] else float('nan')
    star = '***' if r['q'] < 0.05 else ('*' if r['p'] < 0.05 else '')
    print(f'{r["metric"]:26s} {r["mean_tail"]:12.1f} {r["mean_rest"]:12.1f} '
          f'{r["med_tail"]:10.1f} {r["med_rest"]:10.1f} {ratio:7.2f} '
          f'{r["delta"]:8.2f} {r["p"]:9.4f} {r["q"]:8.3f} {star}')

json.dump(res, open('/tmp/fe_tail/stats_out.json', 'w'), indent=1, default=float)

print('\n=== 尾部实例逐项数值 ===')
show = [r['metric'] for r in res[:10]]
hdr = 'instance'.ljust(32) + ''.join(s[:14].rjust(15) for s in show)
print(hdr)
for i in TAIL:
    print(i.ljust(32) + ''.join(f'{by[i].get(s)}'.rjust(15) for s in show))
