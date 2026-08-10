#!/bin/sh
# Post-deploy smoke test for the Liu playground. Usage:
#   ./web/smoke.sh https://liu-playground.fly.dev [password]
# Checks the page version, the example gallery endpoint, and one real
# streamed run (a fast no-training program). Needs only curl + python3.
set -e
BASE="${1:?usage: smoke.sh <base-url> [password]}"
AUTH=""
[ -n "$2" ] && AUTH="-u liu:$2"

say() { printf '%-28s %s\n' "$1" "$2"; }

code=$(curl -s -o /tmp/liu_smoke_page.html -w '%{http_code}' $AUTH "$BASE/")
say "GET /" "$code"
[ "$code" = 200 ] || { echo "  (401 without password means LIU_PASSWORD is set — pass it as arg 2)"; exit 1; }
grep -q 'editor v3' /tmp/liu_smoke_page.html \
  && say "editor version" "v3 single-layer ok" \
  || { say "editor version" "STALE PAGE (no v3 badge)"; exit 1; }

n=$(curl -s $AUTH "$BASE/examples" | python3 -c "import json,sys;print(len(json.load(sys.stdin)))")
say "GET /examples" "$n examples"
[ "$n" -ge 5 ] || { echo "  expected >= 5"; exit 1; }

printf 'seed 0\nY = gaussian([0, 0], 1) ~ 200 via svgd(kernel=rbf, steps=40, lr=0.8)\nplot Y\n' \
  > /tmp/liu_smoke_prog
out=$(curl -s $AUTH -X POST --data-binary @/tmp/liu_smoke_prog "$BASE/run")
echo "$out" | grep -q '"type": "loss"' && say "loss events" "streamed ok" || say "loss events" "none (svgd has no training loss — fine)"
echo "$out" | grep -q '"type": "plot"' && say "plot events" "ok" || { say "plot events" "MISSING"; exit 1; }
echo "$out" | tail -1 | grep -q '"code": 0' && say "run exit" "done, code 0" || { say "run exit" "FAILED: $(echo "$out" | tail -1)"; exit 1; }

echo "SMOKE: PASS"
