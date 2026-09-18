#!/usr/bin/env bash
# The pre-merge gate is author-run (CI cannot install the 4-18 GB package envs); the PR body
# must record it. Reads the body from $PR_BODY. Locally: PR_BODY="$(gh pr view --json body -q .body)" pixi run -e ci ci-gate-line
set -euo pipefail
if ! grep -q '^Gate:' <<< "${PR_BODY:-}"; then
  echo '::error::Run `pixi run -e <package>-dev gate` for each affected package before merge, then add a PR body line: Gate: <package>-dev gate on <machine> @ <commit>'
  exit 1
fi
echo "Gate line present."
