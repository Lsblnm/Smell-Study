#!/usr/bin/env python3
"""为每个 FE 实例计算与语言无关的传统静态度量（不含 ATFD/LAA/FDP）。"""
import json, os, re, subprocess, sys, collections

ROOT = '/data1/zhangyixiang/HuaWei/SWE-bench-FeatureEnvy-test'
REPOS = '/data1/zhangyixiang/HuaWei/fe_static/repos'
PATCH = f'{ROOT}/smell-patch/FeatureEnvy'
INTRO = f'{ROOT}/smell-patch-intro/FeatureEnvy'

SRC_EXT = {'.java', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.php', '.rb', '.go',
           '.rs', '.c', '.h', '.cc', '.cpp', '.hpp', '.hh', '.cxx', '.py', '.scala', '.kt'}
SKIP_DIR = {'.git', 'node_modules', 'vendor', 'third_party', 'thirdparty', 'dist', 'build',
            'target', '.venv', 'fixtures', '__snapshots__'}

BRANCH = re.compile(r'\b(if|else if|elif|for|foreach|while|case|catch|rescue|when|&&|\|\||\?)\b|\?\s*[^:]*:')
IMPORT_RE = re.compile(
    r'^\s*(import\s|from\s+[\'"\w.]+\s+import|#include|require\s*\(|require\s+[\'"]|'
    r'use\s+[\w:\\]+\s*;|using\s+|package\s+[\w./]+\s*$|extern\s+crate)')


# ---------- 解析注入说明 ----------
def parse_intro(iid):
    p = f'{INTRO}/{iid}-FE.md'
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding='utf-8').read()
    head = txt.split('## 质量因子')[0]
    out = {'target_files': [], 'source_files': [], 'methods': []}
    cur = None
    for line in head.split('\n'):
        s = line.strip()
        m = re.match(r'^-\s*(目标文件|来源文件|目标方法)[：:]\s*(.*)$', s)
        if m:
            cur = m.group(1)
            rest = m.group(2).strip()
            if rest:
                _collect(out, cur, rest)
            continue
        if s.startswith('-') and cur and line.startswith(('  -', '\t-', '    -')):
            _collect(out, cur, s[1:].strip())
            continue
        if s.startswith('- '):
            cur = None
    return out


def _collect(out, cur, text):
    keys = {'目标文件': 'target_files', '来源文件': 'source_files', '目标方法': 'methods'}
    vals = re.findall(r'`([^`]+)`', text)
    if not vals:
        return
    k = keys[cur]
    if k == 'methods':
        for v in vals:
            v = v.split('(')[0].strip()
            name = re.split(r'::|\.|->|\\', v)[-1].strip()
            if re.fullmatch(r'[A-Za-z_]\w*', name):
                out[k].append(name)
    else:
        for v in vals:
            if '/' in v or '.' in v:
                out[k].append(v.strip())


# ---------- 解析 patch ----------
def parse_patch(iid):
    p = f'{PATCH}/{iid}-FE.patch'
    txt = open(p, encoding='utf-8', errors='replace').read()
    files, add, dele, hunks = [], 0, 0, 0
    added_lines, deleted_lines = [], []
    per_file = collections.defaultdict(lambda: [0, 0])
    cur = None
    for line in txt.split('\n'):
        if line.startswith('+++ b/'):
            cur = line[6:].strip()
            files.append(cur)
        elif line.startswith('@@'):
            hunks += 1
        elif line.startswith('+') and not line.startswith('+++'):
            add += 1; added_lines.append(line[1:])
            if cur: per_file[cur][0] += 1
        elif line.startswith('-') and not line.startswith('---'):
            dele += 1; deleted_lines.append(line[1:])
            if cur: per_file[cur][1] += 1
    return dict(files=files, add=add, dele=dele, hunks=hunks,
                added_lines=added_lines, deleted_lines=deleted_lines,
                per_file={k: v for k, v in per_file.items()})


# ---------- 仓库扫描 ----------
_repo_cache = {}


def repo_index(iid):
    if iid in _repo_cache:
        return _repo_cache[iid]
    base = f'{REPOS}/{iid}'
    files = []
    total_loc = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SRC_EXT:
                fp = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(fp) > 2_000_000:
                        continue
                except OSError:
                    continue
                files.append(os.path.relpath(fp, base))
    idx = dict(base=base, files=files)
    _repo_cache.clear()
    _repo_cache[iid] = idx
    return idx


def file_loc(base, rel):
    fp = os.path.join(base, rel)
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def rg_count(base, pattern, word=True, fixed=True):
    """返回 (匹配文件数, 匹配行数)"""
    cmd = ['rg', '--no-messages', '--count-matches', '--no-heading']
    if word: cmd.append('-w')
    if fixed: cmd.append('-F')
    for d in SKIP_DIR:
        cmd += ['-g', f'!{d}/**']
    cmd += ['-g', '!*.min.js', '--', pattern, base]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return (None, None)
    nf = nl = 0
    for line in r.stdout.split('\n'):
        if not line.strip():
            continue
        path, _, cnt = line.rpartition(':')
        if os.path.splitext(path)[1].lower() not in SRC_EXT:
            continue
        try:
            nl += int(cnt)
        except ValueError:
            continue
        nf += 1
    return (nf, nl)


def imports_of(base, rel):
    fp = os.path.join(base, rel)
    if not os.path.exists(fp):
        return None
    n = 0
    try:
        for line in open(fp, encoding='utf-8', errors='replace'):
            if IMPORT_RE.match(line):
                n += 1
    except OSError:
        return None
    return n


def path_distance(a, b):
    pa, pb = a.split('/')[:-1], b.split('/')[:-1]
    i = 0
    while i < min(len(pa), len(pb)) and pa[i] == pb[i]:
        i += 1
    return (len(pa) - i) + (len(pb) - i)


def analyse(iid):
    base = f'{REPOS}/{iid}'
    intro = parse_intro(iid)
    pt = parse_patch(iid)
    idx = repo_index(iid)
    m = {'instance_id': iid}

    # --- 项目规模 ---
    m['repo_src_files'] = len(idx['files'])
    m['repo_loc'] = sum(file_loc(base, f) or 0 for f in idx['files'])

    # --- patch 规模 ---
    m['patch_files'] = len(pt['files'])
    m['patch_add'] = pt['add']
    m['patch_del'] = pt['dele']
    m['patch_churn'] = pt['add'] + pt['dele']
    m['patch_hunks'] = pt['hunks']

    # --- 目标 / 来源文件 ---
    tf = [f for f in intro.get('target_files', []) if os.path.exists(os.path.join(base, f))]
    sf = [f for f in intro.get('source_files', []) if os.path.exists(os.path.join(base, f))]
    if not tf:
        # 回退：patch 中新增行最多的文件
        tf = [max(pt['per_file'], key=lambda k: pt['per_file'][k][0])] if pt['per_file'] else []
    m['n_target_files'] = len(tf)
    m['n_source_files'] = len(sf)
    m['n_involved_files'] = len(set(tf) | set(sf) | set(pt['files']))

    locs_t = [file_loc(base, f) or 0 for f in tf]
    locs_s = [file_loc(base, f) or 0 for f in sf]
    m['loc_target'] = max(locs_t) if locs_t else None
    m['loc_source'] = max(locs_s) if locs_s else None
    all_touched = [f for f in set(pt['files']) if os.path.exists(os.path.join(base, f))]
    locs_a = [file_loc(base, f) or 0 for f in all_touched]
    m['loc_touched_max'] = max(locs_a) if locs_a else None
    m['loc_touched_sum'] = sum(locs_a) if locs_a else None

    # --- 文件级扇出（import 数） ---
    fo_t = [imports_of(base, f) for f in tf if imports_of(base, f) is not None]
    fo_s = [imports_of(base, f) for f in sf if imports_of(base, f) is not None]
    m['fanout_target_file'] = max(fo_t) if fo_t else None
    m['fanout_source_file'] = max(fo_s) if fo_s else None

    # --- 文件级扇入（其他文件引用该模块名） ---
    def fanin_file(rel):
        stem = os.path.splitext(os.path.basename(rel))[0]
        if len(stem) < 3:
            return None
        nf, nl = rg_count(base, stem)
        if nf is None:
            return None
        return max(nf - 1, 0)

    fi_t = [x for x in (fanin_file(f) for f in tf) if x is not None]
    fi_s = [x for x in (fanin_file(f) for f in sf) if x is not None]
    m['fanin_target_file'] = max(fi_t) if fi_t else None
    m['fanin_source_file'] = max(fi_s) if fi_s else None
    m['fanin_files_sum'] = (sum(fi_t) + sum(fi_s)) if (fi_t or fi_s) else None

    # --- 被搬迁 / 新建方法的扇入（调用点） ---
    methods = list(dict.fromkeys(intro.get('methods', [])))
    # 补充：patch 新增行里定义的函数名
    defre = re.compile(r'\b(?:def|function|fn|func|public|private|protected|static|inline)\b[^\n(]*?\b'
                       r'([A-Za-z_]\w{3,})\s*\(')
    for l in pt['added_lines']:
        mm = defre.search(l)
        if mm:
            methods.append(mm.group(1))
    methods = [x for x in dict.fromkeys(methods) if len(x) >= 4][:8]
    m['n_moved_methods'] = len(methods)
    fi_m_files, fi_m_refs = [], []
    for name in methods:
        nf, nl = rg_count(base, name)
        if nf is None:
            continue
        fi_m_files.append(nf); fi_m_refs.append(nl)
    m['fanin_method_files_max'] = max(fi_m_files) if fi_m_files else None
    m['fanin_method_refs_sum'] = sum(fi_m_refs) if fi_m_refs else None

    # --- 迁移代码本身 ---
    added = pt['added_lines']
    code = [l for l in added if l.strip() and not l.strip().startswith(('//', '#', '*', '/*'))]
    m['moved_loc'] = len(code)
    calls = collections.Counter(re.findall(r'\b([A-Za-z_]\w*)\s*\(', '\n'.join(code)))
    kw = {'if', 'for', 'while', 'switch', 'catch', 'return', 'sizeof', 'foreach', 'function', 'fn'}
    m['fanout_moved_calls'] = len([k for k in calls if k not in kw])
    m['cc_moved'] = sum(len(BRANCH.findall(l)) for l in code) + 1
    ind = [len(l) - len(l.lstrip()) for l in code if l.strip()]
    m['nesting_moved'] = (max(ind) // 2) if ind else 0

    # --- 结构距离 ---
    if tf and sf:
        d = max(path_distance(t, s) for t in tf for s in sf)
        m['dir_distance'] = d
        m['cross_toplevel'] = int(any(t.split('/')[0] != s.split('/')[0] for t in tf for s in sf))
        m['same_file_move'] = int(all(t == s for t in tf for s in sf))
    else:
        m['dir_distance'] = 0; m['cross_toplevel'] = 0; m['same_file_move'] = 1
    m['target_depth'] = max((f.count('/') for f in tf), default=0)
    return m


def main():
    ids = [l.strip() for l in open('/tmp/fe_tail/paired_ids.txt') if l.strip()]
    out = []
    for i, iid in enumerate(ids, 1):
        if not os.path.exists(f'{REPOS}/{iid}/.done'):
            print(f'[{i}/{len(ids)}] SKIP(no repo) {iid}', flush=True)
            continue
        try:
            out.append(analyse(iid))
            print(f'[{i}/{len(ids)}] ok {iid}', flush=True)
        except Exception as e:
            print(f'[{i}/{len(ids)}] ERR {iid}: {e}', flush=True)
    json.dump(out, open('/tmp/fe_tail/static_metrics.json', 'w'), indent=1)
    print('wrote', len(out))


if __name__ == '__main__':
    main()
