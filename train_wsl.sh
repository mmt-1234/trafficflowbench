#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for argument in "$@"; do
  if [[ "$argument" == --resume || "$argument" == --resume=* ]]; then
    echo "Resume State/Queue separately with src/run.py and the matching .last.pt checkpoint." >&2
    exit 2
  fi
done
release_root=$(cat data/release_root.txt)
checkpoint_dir=${CHECKPOINT_DIR:-artifacts/v2}
mkdir -p "$checkpoint_dir"
exec 9>>"$checkpoint_dir/.train.lock"
if ! flock -n 9; then
  echo "Training is already running in $checkpoint_dir. Use its existing log or a different CHECKPOINT_DIR." >&2
  exit 2
fi
.venv/bin/python -u src/run.py train-state --release-root "$release_root" --checkpoint "$checkpoint_dir/state.pt" "$@"
.venv/bin/python -u src/run.py train-queue --release-root "$release_root" --checkpoint "$checkpoint_dir/queue.pt" --init-state "$checkpoint_dir/state.pt" "$@"
