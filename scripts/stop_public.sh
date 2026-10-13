#!/usr/bin/env bash
# Stop the public tunnel and the studio server started by scripts/serve_public.sh
RUN="$(cd "$(dirname "$0")/.." && pwd)/run"
for f in tunnel server; do
  if [ -f "$RUN/$f.pid" ]; then
    kill "$(cat "$RUN/$f.pid")" 2>/dev/null && echo "stopped $f"
    rm -f "$RUN/$f.pid"
  fi
done
rm -f "$RUN/public_url.txt"
