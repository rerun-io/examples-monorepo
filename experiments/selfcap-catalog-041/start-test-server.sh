#!/usr/bin/env bash
set -euo pipefail
runtime_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n $(ss -H -ltn 'sport = :51241') ]]; then
  echo 'Port 51241 is already in use; no server started.' >&2
  exit 1
fi
# Run from a host-side terminal. This installs only this isolated experiment.
pixi install --manifest-path "$runtime_root/pixi.toml"
server_binary=$("$runtime_root/.pixi/envs/default/bin/python" -c 'from pathlib import Path; import rerun_cli; print(Path(rerun_cli.__file__).with_name("rerun"))')
systemd-run --user --unit=selfcap-catalog-041-test --collect \
  --property=LimitNOFILE=524288 --property="WorkingDirectory=$runtime_root" \
  "$server_binary" server --port 51241
