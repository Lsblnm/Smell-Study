#!/usr/bin/env python3
"""补充度量：精确调用点扇入、import 扇入、删除分散度、语言、测试耦合。"""
import json, os, re, subprocess, collections
import metrics as M

REPOS = M.REPOS
EXT_LANG = {'.java': 'Java', '.js': 'JS/TS', '.jsx': 'JS/TS', '.ts': 'JS/TS', '.tsx': 'JS/TS',
            '.mjs': 'JS/TS', '.cjs': 'JS/TS', '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go',
            '.rs': 'Rust', '.c': 'C', '.h': 'C', '.cc': 'C++', '.cpp': 'C++', '.hpp': 'C++',
            '.hh': 'C++', '.cxx': 'C++', '.py': 'Python', '.scala': 'Scala', '.kt': 'Kotlin'}


def rg_files(base, pattern, regex=False, extra=()):
    cmd = ['rg', '--no-messages', '-l']
    if not regex:
        cmd.append('-F'); cmd.append('-w')
    for d in M.SKIP_DIR:
        cmd += ['-g', f'!{d}/**']
    cmd += ['-g', '!*.min.js', *extra, '--', pattern, base]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return []
    return [os.path.relpath(p, base) for p in r.stdout.split('\n')
            if p.strip() and os.path.splitext(p)[1].lower() in M.SRC_EXT]


def is_test(rel):
    low = rel.lower()
    return ('test' in low) or ('spec' in low) or ('__tests__' in low)


def analyse2(iid):
    base = f'{REPOS}/{iid}'
    intro = M.parse_intro(iid)
    pt = M.parse_patch(iid)
    m = {'instance_id': iid}

    tf = [f for f in intro.get('target_files', []) if os.path.exists(os.path.join(base, f))]
    sf = [f for f in intro.get('source_files', []) if os.path.exists(os.path.join(base, f))]
    if not tf and pt['per_file']:
        tf = [max(pt['per_file'], key=lambda k: pt['per_file'][k][0])]

    # 语言
    exts = [os.path.splitext(f)[1].lower() for f in (tf + sf + pt['files'])]
    langs = collections.Counter(EXT_LANG.get(e) for e in exts if EXT_LANG.get(e))
    m['lang'] = langs.most_common(1)[0][0] if langs else 'other'

    # 删除分散度
    delf = [f for f, (a, d) in pt['per_file'].items() if d > 0]
    m['del_files'] = len(delf)
    m['del_ratio'] = pt['dele'] / max(pt['add'] + pt['dele'], 1)
    m['del_max_one_file'] = max((d for a, d in pt['per_file'].values()), default=0)

    # 方法级精确扇入（调用点，排除定义所在文件）
    methods = [x for x in dict.fromkeys(intro.get('methods', [])) if len(x) >= 4][:6]
    defs = set(tf) | set(sf)
    cs_files, cs_test = [], []
    for name in methods:
        fs = rg_files(base, rf'\b{re.escape(name)}\s*\(', regex=True)
        outside = [f for f in fs if f not in defs]
        cs_files.append(len(outside))
        cs_test.append(len([f for f in outside if is_test(f)]))
    m['callsites_out_max'] = max(cs_files) if cs_files else 0
    m['callsites_out_sum'] = sum(cs_files) if cs_files else 0
    m['callsites_test_sum'] = sum(cs_test) if cs_test else 0

    # 文件级 import 扇入（仅统计 import/require/include 行）
    def import_fanin(rel):
        stem = os.path.splitext(os.path.basename(rel))[0]
        if len(stem) < 3:
            return 0
        pat = rf'^\s*(import|from|#include|require|use|using|extern).*\b{re.escape(stem)}\b'
        return len([f for f in rg_files(base, pat, regex=True) if f != rel])

    m['import_fanin_target'] = max((import_fanin(f) for f in tf), default=0)
    m['import_fanin_source'] = max((import_fanin(f) for f in sf), default=0)

    # 目标 / 来源文件被测试引用数
    def test_refs(rel):
        stem = os.path.splitext(os.path.basename(rel))[0]
        if len(stem) < 3:
            return 0
        return len([f for f in rg_files(base, stem) if is_test(f) and f != rel])

    m['test_refs_target'] = max((test_refs(f) for f in tf), default=0)
    m['test_refs_source'] = max((test_refs(f) for f in sf), default=0)

    # 目标文件内的函数/方法数量（NOM 近似）
    def nom(rel):
        fp = os.path.join(base, rel)
        if not os.path.exists(fp):
            return 0
        txt = open(fp, encoding='utf-8', errors='replace').read()
        return len(re.findall(
            r'^\s*(?:(?:public|private|protected|static|final|inline|const|async|export)\s+)*'
            r'(?:def|function|fn|func|sub)?\s*[\w<>\[\]:*&~ ]*?\b[A-Za-z_]\w*\s*\([^;]*\)\s*(?:const\s*)?\{',
            txt, re.M))

    m['nom_target'] = max((nom(f) for f in tf), default=0)
    m['nom_source'] = max((nom(f) for f in sf), default=0)
    return m


if __name__ == '__main__':
    ids = [l.strip() for l in open('/tmp/fe_tail/paired_ids.txt') if l.strip()]
    out = []
    for i, iid in enumerate(ids, 1):
        try:
            out.append(analyse2(iid)); print(f'[{i}/{len(ids)}] ok {iid}', flush=True)
        except Exception as e:
            print(f'[{i}/{len(ids)}] ERR {iid}: {e}', flush=True)
    json.dump(out, open('/tmp/fe_tail/static_metrics2.json', 'w'), indent=1)
