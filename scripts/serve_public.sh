#!/usr/bin/env bash
# Host the live studio for the public:
#   1) server.py with PUBLIC=1 limits, bound to 127.0.0.1 (jobs go to any idle GPU; queue when none is free)
#   2) a Cloudflare quick tunnel (outbound only, no root/firewall changes) -> https://<random>.trycloudflare.com
# The URL changes whenever the tunnel restarts; it is written to run/public_url.txt. Stop with scripts/stop_public.sh.
# Optional env: GPUS=4,5,6,7 (restrict the pool)  MAX_PARALLEL=n  PER_IP_TOTAL=1  DEFAULT_GUIDANCE=mvdream  PORT=8549
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/run"; mkdir -p "$RUN"
PORT="${PORT:-8549}"
BIN="$HOME/bin/cloudflared"

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dreamgaussian

# --- server
if [ -f "$RUN/server.pid" ] && kill -0 "$(cat "$RUN/server.pid")" 2>/dev/null; then
  echo "server already running (pid $(cat "$RUN/server.pid"))"
else
  for p in $(pgrep -f "python server.py --port $PORT" || true); do kill "$p" || true; done
  sleep 1
  cd "$ROOT"
  PUBLIC=1 DEFAULT_GUIDANCE="${DEFAULT_GUIDANCE:-mvdream}" setsid nohup python server.py --host 127.0.0.1 --port "$PORT" > "$RUN/server.log" 2>&1 < /dev/null &
  echo $! > "$RUN/server.pid"
  for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/api/config" > /dev/null && break; sleep 1; done
  echo "server up on 127.0.0.1:$PORT (pid $(cat "$RUN/server.pid")): $(curl -s "http://127.0.0.1:$PORT/api/queue")"
fi

# --- tunnel
if [ ! -x "$BIN" ]; then
  mkdir -p "$(dirname "$BIN")"
  curl -fsSL -o "$BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$BIN"
fi
"$BIN" --version
if [ -f "$RUN/tunnel.pid" ] && kill -0 "$(cat "$RUN/tunnel.pid")" 2>/dev/null; then
  echo "tunnel already running (pid $(cat "$RUN/tunnel.pid"))"
else
  setsid nohup "$BIN" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$RUN/tunnel.log" 2>&1 < /dev/null &
  echo $! > "$RUN/tunnel.pid"
  for _ in $(seq 1 45); do grep -qo 'https://[a-z0-9-]*\.trycloudflare\.com' "$RUN/tunnel.log" && break; sleep 1; done
fi
URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$RUN/tunnel.log" | head -1 || true)
echo "$URL" > "$RUN/public_url.txt"
if [ -n "$URL" ]; then echo "PUBLIC URL: $URL"; else echo "tunnel URL not found yet - see $RUN/tunnel.log"; fi
