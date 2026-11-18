#!/usr/bin/env bash
# 统一启动入口。用法：
#   ./run.sh [prob|embed] [--max-evaluations 30 --max-cost 15] [extra args]
# 环境变量来自 .env（DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL）；官方代理只需改 BASE_URL。
set -euo pipefail
cd "$(dirname "$0")"
BASELINE="${1:-}"
if [[ "$BASELINE" == "prob" || "$BASELINE" == "embed" ]]; then shift; else BASELINE=""; fi
PY="${PYTHON:-python3}"
if [[ -n "$BASELINE" ]]; then
  exec "$PY" -m harness.loop --baseline "$BASELINE" "$@"
else
  exec "$PY" -m harness.loop "$@"
fi
