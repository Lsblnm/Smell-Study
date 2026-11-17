import json, glob, os, re, statistics, collections

ROOT = '/data1/zhangyixiang/HuaWei/SWE-bench-FeatureEnvy-test/traj_results/opencode/FE'
RECON = {'read', 'grep', 'glob'}
RO_BASH = re.compile(r'\s*(ls|cat|find|grep|rg|head|tail|wc|sed -n|tree|git (log|show|diff|status))\b')


def parse(path):
    recon = tools = 0
    tok = 0
    last_total = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get('type')
        if t == 'tool_use':
            tools += 1
            tl = o['part'].get('tool')
            if tl in RECON:
                recon += 1
            elif tl == 'bash':
                cmd = (o['part'].get('state', {}).get('input') or {}).get('command', '')
                if RO_BASH.match(cmd):
                    recon += 1
        elif t == 'step_finish':
            tk = o['part'].get('tokens') or {}
            tok += tk.get('total', 0)
    return recon, tools, tok


rows = collections.defaultdict(dict)
for run in ['1', '2', '3']:
    for cond in ['none', 'smell']:
        base = f'{ROOT}/{run}/{cond}'
        if not os.path.isdir(base):
            continue
        for iid in os.listdir(base):
            d = os.path.join(base, iid)
            tj = os.path.join(d, 'trajectory.jsonl')
            rj = os.path.join(d, 'result.json')
            if not os.path.exists(tj):
                continue
            recon, tools, tok = parse(tj)
            el = None
            if os.path.exists(rj):
                try:
                    rec = json.load(open(rj))
                    el = rec.get('agent_seconds', rec.get('elapsed_seconds'))
                except Exception:
                    pass
            rows[iid][(run, cond)] = dict(recon=recon, tools=tools, tok=tok, elapsed=el)

json.dump({k: {f'{r}|{c}': v for (r, c), v in d.items()} for k, d in rows.items()},
          open('/tmp/fe_tail/traj_metrics.json', 'w'), indent=1)

print('instances:', len(rows))
runs = collections.Counter()
for k, d in rows.items():
    runs[tuple(sorted(d))] += 1
for k, v in runs.items():
    print(v, k)
