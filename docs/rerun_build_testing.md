# Testing Rerun builds under Pixi

Most of the time a package depends on the **released** `rerun-sdk`. Occasionally we need to test a
Rerun build that isn't released yet — to pick up a fix that's merged (or still on a PR branch)
upstream. This doc captures the three ways to get a Rerun build into a Pixi environment, ordered
from "default" to "last resort". It's monorepo-wide: the `rerun-prerelease` lane is shared (today
`mv-api-catalog` and `simplecv-catalog` use it), so pin/repin it here, not per package.

## 1. Released wheel (the default)

The `common` feature pins the public release:

```toml
[feature.common.pypi-dependencies]
rerun-sdk = "==0.33.0"
```

Nothing to do — normal `pixi run -e <env>` uses it. Use this unless you specifically need an
unreleased fix.

## 2. Commit wheel via `find-links` (the prerelease lane)

Rerun's CI builds a wheel for **every pushed commit** — both `main` and **PR branches** — and
uploads it to `https://build.rerun.io/commit/<short-sha>/wheels/` (the dir is keyed by the 7-char
commit hash). So we can install any commit's build by pointing Pixi's `find-links` at it. We keep a
dedicated **`rerun-prerelease`** feature for this:

```toml
[feature.rerun-prerelease.pypi-dependencies]
rerun-sdk = { version = "==0.34.0a1+dev", extras = ["datafusion", "dataloader"] }

[feature.rerun-prerelease.pypi-options]
index-url = "https://pypi.org/simple"
find-links = [{ url = "https://build.rerun.io/commit/deeb4e6/wheels/" }]
prerelease-mode = "allow"
```

It's consumed by composing it into a `no-default-feature` environment **beside `catalog-common`**
(the rerun-pin-free mirror of `common`) — composing `common` instead would conflict with its
released `rerun-sdk==0.33.0`. See `simplecv-catalog` / `mv-api-catalog` in the root `pixi.toml`.

### Repinning to a new commit

1. **Find the commit.** A merge commit on `main`, or a PR's head sha
   (`gh pr view <N> --repo rerun-io/reality --json headRefOid`).
2. **Check the wheel index first** — don't assume it's there for your platform:
   ```bash
   curl -fsSL https://build.rerun.io/commit/<short-sha>/wheels/
   ```
3. **Update both the hash and the version string** in `[feature.rerun-prerelease.*]` to match the
   wheel filename exactly (e.g. `rerun_sdk-0.34.0a1+dev-...` → `version = "==0.34.0a1+dev"` and
   `find-links = .../commit/<short-sha>/wheels/`). A mismatched version string won't resolve.
4. **Re-lock** on a linux-64 host (`pixi lock`), then `pixi install -e <env>`.

### Verifying the install

```bash
pixi run -e <env> python -c "import rerun; print(rerun.__version__)"   # -> 0.34.0-alpha.1+dev
```
The version string alone confirms you're on the prerelease wheel (a released rerun prints e.g.
`0.33.0`). For extra confidence that you got *that specific* build, exercise the behavior the target
change introduced — e.g. for #2496, time a `register()` of a large recording and confirm it no longer
stalls (footer-first enumeration makes a multi-GB RRD register in well under a second).

### When this **doesn't** work (and you fall back to method 3)

- **Missing platform.** PR-branch commits typically publish only **linux-x86_64**; the full matrix
  (aarch64, macOS) is built on merge/release. If you're on a platform that commit didn't build,
  there's no wheel for you. (Step 2's `curl` shows exactly which wheels exist.)
- **No wheel at all.** CI is still running (wait a bit), failed/was skipped, the PR is a draft/fork
  with no CI, or the commit is old enough that its build was pruned from storage.
- **Local, uncommitted Rerun changes.** No pushed commit hash exists to point at.

### Worked example — `rerun-io/reality#2496` (fast OSS-catalog register)

```bash
gh pr view 2496 --repo rerun-io/reality --json headRefOid   # -> deeb4e6...
curl -fsSL https://build.rerun.io/commit/deeb4e6/wheels/     # -> rerun_sdk-0.34.0a1+dev-...x86_64.whl
```
This is an **open** PR (not merged), yet its branch-tip commit has a CI wheel that already contains
the fix — proof that method 2 works for unmerged branches. It's linux-x86_64 only, which is fine for
the linux-64 `simplecv-catalog` env that uses it.

## 3. Build from a branch locally (last resort)

Only when method 2 can't help: you need a platform CI didn't build, there's no CI wheel, or you have
local uncommitted Rerun changes. Build the wheel from the reality clone and point `find-links` at it:

```bash
cd /home/pablo/0Dev/work/rerun-projects/reality
git fetch origin <branch> && git checkout <branch>   # e.g. nick/slow_register for #2496
cd rerun && pixi run py-build-wheel                   # see rerun/BUILD.md; lands in dist/<target>/
```
Then in the feature's `pypi-options`, set
`find-links = [{ url = "file:///home/pablo/0Dev/work/rerun-projects/reality/rerun/dist/<target>/" }]`
with the matching version string. (`py-build-wheels-sdk-only` omits the viewer; use `py-build-wheel`
for parity.)
