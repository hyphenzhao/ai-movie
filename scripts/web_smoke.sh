#!/usr/bin/env bash
# Smoke test for the web UI API.  Usage: bash scripts/web_smoke.sh [http://host:8000] [project]
set -u
BASE="${1:-http://127.0.0.1:8000}"
PROJ="${2:-output_test}"
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
j() { python3 -c "import sys,json; d=json.load(sys.stdin); print($1)"; }

curl -sf "$BASE/api/health" | grep -q '"ok":true' && ok health || bad health
curl -sf "$BASE/api/projects" | grep -q "\"$PROJ\"" && ok "projects lists $PROJ" || bad "projects lists $PROJ"
S=$(curl -sf "$BASE/api/projects/$PROJ" | j "' '.join(sorted(set(v['status'] for v in d['steps'].values())))")
echo "     statuses: $S"
echo "$S" | grep -qE "done|ready|stale|locked" && ok "step statuses" || bad "step statuses"
A=$(curl -sf "$BASE/api/projects/$PROJ/segments" | j "d[0].get('audio_fit_url') or d[0].get('audio_url') or ''")
[ -n "$A" ] && ok "segments have audio urls" || bad "segments audio urls"
CODE=$(curl -s -r 0-99 -o /dev/null -w "%{http_code}" "$BASE$A")
[ "$CODE" = "206" ] && ok "range request 206" || bad "range request ($CODE)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/api/media?p=/etc/passwd")
[ "$CODE" = "403" ] && ok "path traversal blocked" || bad "path traversal ($CODE)"
curl -s -N --max-time 18 "$BASE/api/events" | grep -q "ping" && ok "global SSE ping" || bad "global SSE ping"
D=$(curl -sf "$BASE/api/projects/$PROJ/deliverables" | j "next((f['download'] for f in d if f['download']), '')")
if [ -n "$D" ]; then
  curl -s -o /dev/null -D - -r 0-10 "$BASE$D" | grep -qi "content-disposition: attachment" && ok "download header" || bad "download header"
fi
curl -sf "$BASE/" | grep -q "app.js" && ok "index page" || bad "index page"
exit $fail
