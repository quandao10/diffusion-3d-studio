#!/usr/bin/env bash
# Gallery runs (sequential, same GPU). usage: GPU=4 bash scripts/queue_gallery.sh
cd "$(dirname "$0")/.."
run() { bash scripts/run_text_to_3d.sh "$1" "$2" "${3:-0}" > /dev/null 2>&1; echo "$2 exit=$?"; }
run "a ripe pineapple" pineapple 0
run "a wooden treasure chest" treasure_chest 0
run "a cute corgi puppy" corgi 0
