# Model Architecture Source Links Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish model architecture manifests whose code links point at a valid `phi9t/megatronlm` source revision.

**Architecture:** Keep block `ref` values as repo-relative `file:line` strings and fix only the manifest-level source revision. `build_model_arch.py` will read `MEGATRON_EXPLORER_SOURCE_REF`, default to `main`, and the existing Pages workflow will continue passing `${{ github.sha }}` so live links are pinned to the deployed commit.

**Tech Stack:** Python manifest generators, Vite/React static explorer, GitHub Pages workflow, shell verification with `curl` and `gh`.

---

## File Structure

- Modify `explorer/scripts/build_model_arch.py`: replace the hardcoded `MODEL_SOURCE_REF` with an environment-driven value.
- Modify `explorer/scripts/verify_manifests.py`: add a local structural check that model `source` is a non-empty string and that generated refs remain repo-local and line-valid.
- Modify `explorer/public/data/models/*.json`: regenerate stable checked-in data with `source: "main"`.
- Use existing `.github/workflows/deploy-megatron-explorer-pages.yml`: confirm it already passes `MEGATRON_EXPLORER_SOURCE_REF: ${{ github.sha }}` and no code change is needed unless verification shows otherwise.

## Task 1: Make Model Source Revision Configurable

**Files:**
- Modify: `explorer/scripts/build_model_arch.py`
- Modify: `explorer/scripts/verify_manifests.py`
- Modify generated data: `explorer/public/data/models/*.json`

- [ ] **Step 1: Write the failing source override check**

Run this command before changing implementation:

```bash
MEGATRON_EXPLORER_SOURCE_REF=abc123 python3 explorer/scripts/build_model_arch.py \
  --repo-root . \
  --out-dir /tmp/megatron-model-source-check
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('/tmp/megatron-model-source-check/qwen3-8b.json').read_text())
assert data['source'] == 'abc123', data['source']
PY
```

Expected: FAIL with `AssertionError: d681f6898b64d613991e0b1ac4b2c631bf9aca36`.

- [ ] **Step 2: Implement environment-driven source**

In `explorer/scripts/build_model_arch.py`, import `os` and replace the current constant:

```python
import os
```

```python
MODEL_SOURCE_REF = os.environ.get("MEGATRON_EXPLORER_SOURCE_REF", "main")
```

Keep `build_manifest()` unchanged where it writes:

```python
"source": MODEL_SOURCE_REF,
```

- [ ] **Step 3: Tighten manifest source validation**

In `explorer/scripts/verify_manifests.py`, inside `check_model()`, after the required-field loop, add:

```python
    if not isinstance(data.get("source"), str) or not data.get("source", "").strip():
        errors.append(f"{path}: model source must be a non-empty string")
```

This catches missing/empty revisions while keeping local verification offline.

- [ ] **Step 4: Verify the override now passes**

Run:

```bash
rm -rf /tmp/megatron-model-source-check
MEGATRON_EXPLORER_SOURCE_REF=abc123 python3 explorer/scripts/build_model_arch.py \
  --repo-root . \
  --out-dir /tmp/megatron-model-source-check
python3 - <<'PY'
import json
from pathlib import Path
for path in Path('/tmp/megatron-model-source-check').glob('*.json'):
    if path.name == 'index.json':
        continue
    data = json.loads(path.read_text())
    assert data['source'] == 'abc123', (path.name, data['source'])
print('source override passed')
PY
```

Expected: prints `source override passed`.

- [ ] **Step 5: Regenerate stable checked-in model data**

Run:

```bash
./explorer/scripts/workflow.sh gen-data
```

Expected: model JSON files are regenerated and `explorer/public/data/models/*.json` now contain:

```json
"source": "main"
```

- [ ] **Step 6: Run local verification**

Run:

```bash
cd explorer
./scripts/workflow.sh verify
npm run lint
```

Expected:
- `workflow.sh verify` exits 0.
- `npm run lint` exits 0. Existing warnings are acceptable; errors are not.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add explorer/scripts/build_model_arch.py explorer/scripts/verify_manifests.py explorer/public/data/models
git commit -m "fix: parameterize model architecture source revision"
```

## Task 2: Verify Published Link Path

**Files:**
- Read: `.github/workflows/deploy-megatron-explorer-pages.yml`
- Read: `explorer/dist/**`
- No source edits expected unless the workflow does not pass `MEGATRON_EXPLORER_SOURCE_REF`.

- [ ] **Step 1: Confirm workflow passes the source SHA**

Run:

```bash
rg "MEGATRON_EXPLORER_SOURCE_REF|VITE_SOURCE_REF" .github/workflows/deploy-megatron-explorer-pages.yml
```

Expected output includes:

```yaml
MEGATRON_EXPLORER_SOURCE_REF: ${{ github.sha }}
VITE_SOURCE_REF: ${{ github.sha }}
```

- [ ] **Step 2: Build a Pages artifact with the current commit SHA**

Run:

```bash
cd explorer
MEGATRON_EXPLORER_REPO_HOME=https://github.com/phi9t/megatronlm \
MEGATRON_EXPLORER_SOURCE_REF="$(git -C .. rev-parse HEAD)" \
VITE_BASE_PATH=/megatronlm/explorer/ \
VITE_REPO_HOME=https://github.com/phi9t/megatronlm \
VITE_SOURCE_REF="$(git -C .. rev-parse HEAD)" \
./scripts/workflow.sh verify
```

Expected: exits 0.

- [ ] **Step 3: Check generated dist model source**

Run:

```bash
python3 - <<'PY'
import json
import subprocess
from pathlib import Path
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
for path in Path('explorer/dist/data/models').glob('*.json'):
    if path.name == 'index.json':
        continue
    data = json.loads(path.read_text())
    assert data['source'] == head, (path.name, data['source'], head)
print('dist model sources match HEAD')
PY
```

Expected: prints `dist model sources match HEAD`.

- [ ] **Step 4: Check sample GitHub source links resolve**

Run:

```bash
HEAD_SHA="$(git rev-parse HEAD)"
for url in \
  "https://github.com/phi9t/megatronlm/blob/${HEAD_SHA}/megatron/core/models/gpt/gpt_model.py#L154" \
  "https://github.com/phi9t/megatronlm/blob/${HEAD_SHA}/megatron/core/transformer/attention.py#L1397" \
  "https://github.com/phi9t/megatronlm/blob/${HEAD_SHA}/megatron/core/transformer/multi_latent_attention.py#L527"
do
  code="$(curl -I -L -s -o /dev/null -w '%{http_code}' "$url")"
  test "$code" = "200" || { echo "$code $url"; exit 1; }
done
echo "sample source links resolve"
```

Expected: prints `sample source links resolve`.

- [ ] **Step 5: Restore stable generated data**

Run:

```bash
./explorer/scripts/workflow.sh gen-data
git status --short
```

Expected: no unwanted `dist` changes are tracked; checked-in model JSON remains at `source: "main"`.

- [ ] **Step 6: Commit any Task 2 changes**

If Task 2 required workflow edits, commit them:

```bash
git add .github/workflows/deploy-megatron-explorer-pages.yml
git commit -m "fix: pin explorer pages model links to deploy sha"
```

If no files changed in Task 2, do not create an empty commit.

## Task 3: Publish and Live Verify

**Files:**
- No source edits expected.
- Publish branch: `feature/megatron-explorer`
- Publish Pages branch: `gh-pages`

- [ ] **Step 1: Push the feature branch**

Run:

```bash
git push
```

Expected: branch `feature/megatron-explorer` is pushed to `origin`.

- [ ] **Step 2: Wait for Pages workflow**

Run:

```bash
RUN_ID="$(gh run list --workflow "Deploy Megatron Explorer Pages" --branch feature/megatron-explorer --limit 1 --json databaseId --jq '.[0].databaseId')"
gh run watch "$RUN_ID" --exit-status
```

Expected: run exits 0.

- [ ] **Step 3: Poll GitHub Pages until the new artifact is live**

Run:

```bash
HEAD_SHA="$(git rev-parse HEAD)"
for i in $(seq 1 30); do
  pg_status="$(gh api repos/phi9t/megatronlm/pages --jq .status 2>/dev/null || echo unknown)"
  html="$(curl -fsS -L https://phi9t.github.io/megatronlm/explorer/ || true)"
  asset="$(printf '%s' "$html" | sed -n 's/.*src="\([^"]*index-[^"]*\.js\)".*/\1/p' | head -1)"
  if [ -n "$asset" ]; then
    curl -fsS -L "https://phi9t.github.io${asset}" -o /tmp/megatron-explorer-live.js || true
    sha_count="$(grep -c "$HEAD_SHA" /tmp/megatron-explorer-live.js 2>/dev/null || true)"
    upstream_count="$(grep -c 'github.com/NVIDIA/Megatron-LM' /tmp/megatron-explorer-live.js 2>/dev/null || true)"
    echo "$i status=$pg_status asset=$asset sha_count=$sha_count upstream_count=$upstream_count"
    if [ "$pg_status" = "built" ] && [ "$sha_count" != "0" ] && [ "$upstream_count" = "0" ]; then
      exit 0
    fi
  fi
  sleep 5
done
exit 1
```

Expected: exits 0 with `sha_count` non-zero and `upstream_count=0`.

- [ ] **Step 4: Verify live model JSON source**

Run:

```bash
HEAD_SHA="$(git rev-parse HEAD)"
curl -fsS -L https://phi9t.github.io/megatronlm/explorer/data/models/qwen3-8b.json -o /tmp/qwen3-8b-live.json
python3 - <<'PY'
import json
import os
import subprocess
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
data = json.load(open('/tmp/qwen3-8b-live.json'))
assert data['source'] == head, (data['source'], head)
print('live model source matches HEAD')
PY
```

Expected: prints `live model source matches HEAD`.

- [ ] **Step 5: Verify live model-architecture sample links**

Run:

```bash
HEAD_SHA="$(git rev-parse HEAD)"
for url in \
  "https://github.com/phi9t/megatronlm/blob/${HEAD_SHA}/megatron/core/models/gpt/gpt_model.py#L154" \
  "https://github.com/phi9t/megatronlm/blob/${HEAD_SHA}/megatron/core/transformer/attention.py#L1397" \
  "https://github.com/phi9t/megatronlm/blob/${HEAD_SHA}/megatron/core/transformer/multi_latent_attention.py#L527"
do
  code="$(curl -I -L -s -o /dev/null -w '%{http_code}' "$url")"
  test "$code" = "200" || { echo "$code $url"; exit 1; }
done
echo "live source links resolve"
```

Expected: prints `live source links resolve`.
