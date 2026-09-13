#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Records the reference scenario with asciinema.
#
# The scenario itself needs nothing but Python: the models are scripted, the
# utility's systems are in process, and the hub runs on loopback. asciinema is
# the only extra tool, and it is only needed to make the recording.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
out="${1:-$root/docs/demo.cast}"

if ! command -v asciinema >/dev/null 2>&1; then
    echo "asciinema is not installed. On macOS: brew install asciinema" >&2
    exit 1
fi

python="$root/.venv/bin/python"
if [ ! -x "$python" ]; then
    echo "no virtualenv at $root/.venv. Run: make install" >&2
    exit 1
fi

mkdir -p "$(dirname "$out")"
cd "$root"
asciinema rec "$out" --overwrite --idle-time-limit 1 \
    --title "Dead Reckoning: an outage that takes out the link to headquarters" \
    --command "$python -m demo.scenario_outage"

echo "recorded to $out"
