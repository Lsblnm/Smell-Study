#!/usr/bin/env bash
# 从本地 SWE-bench 镜像导出 /testbed 源码（base commit 状态）
set -u
OUT=/data1/zhangyixiang/HuaWei/fe_static/repos
mkdir -p "$OUT"
IDS_FILE=/tmp/fe_tail/paired_ids.txt

export_one() {
  iid="$1"
  dest="$OUT/$iid"
  if [ -f "$dest/.done" ]; then echo "skip $iid"; return 0; fi
  rm -rf "$dest"; mkdir -p "$dest"
  img="docker.io/swebench/sweb.eval.x86_64.$(echo "$iid" | sed 's/__/_1776_/' | tr '[:upper:]' '[:lower:]'):latest"
  docker run --rm --entrypoint tar "$img" \
      -cf - -C / \
      --exclude='testbed/.git' \
      --exclude='testbed/node_modules' \
      --exclude='*/node_modules' \
      --exclude='testbed/target/debug' \
      --exclude='testbed/target/release' \
      --exclude='testbed/.venv' \
      --exclude='testbed/vendor/bundle' \
      testbed 2>/dev/null | tar -xf - -C "$dest" --strip-components=1 2>/dev/null
  if [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    touch "$dest/.done"; echo "ok $iid $(du -sh "$dest" | cut -f1)"
  else
    echo "FAIL $iid"
  fi
}
export -f export_one
export OUT

xargs -a "$IDS_FILE" -P 4 -I{} bash -c 'export_one "$@"' _ {}
echo "ALL DONE"
du -sh "$OUT"
