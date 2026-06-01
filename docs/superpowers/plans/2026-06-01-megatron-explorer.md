# Megatron Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `megatronlm/explorer` Vite/React app that visualizes Megatron-LM internals through Guide, Components, Parallelism, and Model Architecture modes.

**Architecture:** Copy/adapt the proven `~/CodeBase/vllm/explorer` shell and shared kit into `megatronlm/explorer`, then replace vLLM-specific data and views with Megatron-specific static manifests. Python generators produce repo-relative JSON/Markdown under `explorer/public/data`; the SPA is a pure static reader with local React state only.

**Tech Stack:** React 19, TypeScript, Vite, Tailwind 4, lucide-react, react-markdown, Mermaid, Radix Slider, Python standard library generators, npm scripts.

---

## Reference Material

- Design spec: `docs/superpowers/specs/2026-06-01-megatron-explorer-design.md`
- Source explorer to adapt: `/Users/phi9t/CodeBase/vllm/explorer`
- Megatron docs:
  - `README.md`
  - `docs/get-started/overview.md`
  - `docs/user-guide/parallelism-guide.md`
  - `docs/ultra-scale-playbook/megatron-mapping.md`
  - `docs/user-guide/features/*.md`
- Megatron source anchors:
  - `examples/llama/llama_config.py`
  - `examples/llama/run_llama.py`
  - `megatron/training/training.py`
  - `megatron/training/initialize.py`
  - `megatron/training/yaml_arguments.py`
  - `megatron/core/models/gpt/gpt_model.py`
  - `megatron/core/transformer/transformer_block.py`
  - `megatron/core/transformer/transformer_layer.py`
  - `megatron/core/transformer/attention.py`
  - `megatron/core/transformer/mlp.py`
  - `megatron/core/transformer/moe/*`
  - `megatron/core/tensor_parallel/*`
  - `megatron/core/pipeline_parallel/*`
  - `megatron/core/distributed/*`
  - `megatron/core/optimizer/*`

## File Structure

Create `explorer/` as an isolated frontend project. It owns all JS dependencies through `explorer/package.json` and does not modify Megatron's Python dependency metadata.

```text
explorer/
  README.md
  index.html
  package.json
  package-lock.json
  tsconfig.json
  tsconfig.app.json
  tsconfig.node.json
  vite.config.ts
  eslint.config.js
  public/
    favicon.svg
    logo-mark.svg
    data/
      guide.md
      graphs/index.json
      graphs/training-loop.json
      graphs/core-stack.json
      parallelism/index.json
      parallelism/strategies.json
      models/index.json
      models/qwen3-0_6b.json
      models/qwen3-8b.json
      models/qwen3-30b-a3b.json
      models/deepseek-v3.json
  scripts/
    _refs.py
    build_guide.py
    build_component_manifest.py
    build_parallelism_manifest.py
    build_model_arch.py
    verify_manifests.py
    workflow.sh
  src/
    App.tsx
    main.tsx
    index.css
    explorer-kit/
      AsyncBoundary.tsx
      DetailDrawer.tsx
      SubjectSwitcher.tsx
      ViewTabs.tsx
      mode.ts
    lib/
      assets.ts
      fetch.ts
      utils.ts
    guide/
      GuideExplorer.tsx
    components-deepdive/
      ArchitectureGraph.tsx
      ComponentDrawer.tsx
      ComponentExplorer.tsx
      types.ts
    parallelism/
      ParallelismExplorer.tsx
      RankGrid.tsx
      StrategyDiagram.tsx
      types.ts
    architecture/
      ArchitectureExplorer.tsx
      ModelCircuit.tsx
      blockTypes.ts
      modelArch.ts
```

Responsibilities:

- `scripts/_refs.py`: resolve `file:line` anchors using symbol grep.
- `scripts/build_*.py`: emit deterministic static data.
- `scripts/verify_manifests.py`: schema and source-reference smoke checks.
- `scripts/workflow.sh`: orchestration for `gen-data`, `install`, `build`, `verify`, `dev`.
- `src/explorer-kit/*`: shared UI primitives copied/adapted from vLLM explorer.
- `src/lib/*`: portable asset/data URLs, GitHub source links, utility functions.
- Mode directories: one generic renderer per manifest family.

---

### Task 1: Scaffold The Explorer Project

**Files:**
- Create: `explorer/package.json`
- Create: `explorer/package-lock.json`
- Create: `explorer/index.html`
- Create: `explorer/tsconfig.json`
- Create: `explorer/tsconfig.app.json`
- Create: `explorer/tsconfig.node.json`
- Create: `explorer/vite.config.ts`
- Create: `explorer/eslint.config.js`
- Create: `explorer/public/favicon.svg`
- Create: `explorer/public/logo-mark.svg`
- Create: `explorer/src/main.tsx`
- Create: `explorer/src/index.css`
- Create: `explorer/src/App.tsx`
- Create: `explorer/README.md`

- [ ] **Step 1: Copy the baseline Vite files from the vLLM explorer**

Run:

```bash
mkdir -p explorer
cp /Users/phi9t/CodeBase/vllm/explorer/package.json explorer/package.json
cp /Users/phi9t/CodeBase/vllm/explorer/package-lock.json explorer/package-lock.json
cp /Users/phi9t/CodeBase/vllm/explorer/index.html explorer/index.html
cp /Users/phi9t/CodeBase/vllm/explorer/tsconfig.json explorer/tsconfig.json
cp /Users/phi9t/CodeBase/vllm/explorer/tsconfig.app.json explorer/tsconfig.app.json
cp /Users/phi9t/CodeBase/vllm/explorer/tsconfig.node.json explorer/tsconfig.node.json
cp /Users/phi9t/CodeBase/vllm/explorer/vite.config.ts explorer/vite.config.ts
cp /Users/phi9t/CodeBase/vllm/explorer/eslint.config.js explorer/eslint.config.js
mkdir -p explorer/public explorer/src
cp /Users/phi9t/CodeBase/vllm/explorer/public/favicon.svg explorer/public/favicon.svg
cp /Users/phi9t/CodeBase/vllm/explorer/public/logo-mark.svg explorer/public/logo-mark.svg
cp /Users/phi9t/CodeBase/vllm/explorer/src/main.tsx explorer/src/main.tsx
cp /Users/phi9t/CodeBase/vllm/explorer/src/index.css explorer/src/index.css
```

Expected: files exist under `explorer/`.

- [ ] **Step 2: Rename the package and page metadata**

Edit `explorer/package.json`:

```json
{
  "name": "megatron-explorer",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "npm run typecheck && vite build",
    "lint": "eslint .",
    "typecheck": "tsc -b",
    "preview": "vite preview"
  }
}
```

Keep the dependency and devDependency blocks copied from vLLM.

Edit `explorer/index.html` so the title is:

```html
<title>Megatron Explorer</title>
```

- [ ] **Step 3: Create the initial App shell**

Create `explorer/src/App.tsx`:

```tsx
import { useState } from 'react'
import { ArrowLeft, BookOpen, Boxes, Network, SplitSquareHorizontal } from 'lucide-react'
import { REPO_HOME, logoMarkUrl } from './lib/assets'
import type { ExplorerMode } from './explorer-kit/mode'
import GuideExplorer from './guide/GuideExplorer'
import ComponentExplorer from './components-deepdive/ComponentExplorer'
import ParallelismExplorer from './parallelism/ParallelismExplorer'
import ArchitectureExplorer from './architecture/ArchitectureExplorer'

const MODES: ExplorerMode[] = [
  {
    id: 'guide',
    label: 'Guide',
    icon: BookOpen,
    subtitle: 'Megatron-LM and Megatron Core, code-first',
    View: GuideExplorer,
  },
  {
    id: 'components',
    label: 'Components',
    icon: Network,
    subtitle: 'Training loop and Megatron Core subsystem graphs',
    View: ComponentExplorer,
  },
  {
    id: 'parallelism',
    label: 'Parallelism',
    icon: SplitSquareHorizontal,
    subtitle: 'TP, PP, DP, CP, EP, and FSDP rank geometry',
    View: ParallelismExplorer,
  },
  {
    id: 'architecture',
    label: 'Model Architecture',
    icon: Boxes,
    subtitle: 'Qwen3 and DeepSeek-V3 circuits grounded in Megatron Core',
    View: ArchitectureExplorer,
  },
]

export default function App() {
  const [activeId, setActiveId] = useState<string>(MODES[0].id)
  const active = MODES.find((mode) => mode.id === activeId) ?? MODES[0]
  const ActiveView = active.View

  return (
    <div className="relative min-h-screen">
      <div className="observatory-bg" aria-hidden="true" />
      <div className="explorer-container">
        <header className="explorer-header">
          <div>
            <a href="#main-content" className="skip-link">
              Skip to main content
            </a>
            <a href={REPO_HOME} className="back-home-link" target="_blank" rel="noopener noreferrer">
              <ArrowLeft size={14} aria-hidden="true" />
              <span>NVIDIA/Megatron-LM</span>
            </a>
            <div className="header-title-row">
              <img src={logoMarkUrl()} alt="" className="header-logo" width={32} height={32} />
              <h1>Megatron Explorer</h1>
            </div>
            <p>{active.subtitle}</p>
          </div>
          <nav className="family-switch" aria-label="Explorer section">
            {MODES.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                className={`family-switch-btn ${activeId === id ? 'active' : ''}`}
                aria-pressed={activeId === id}
                onClick={() => setActiveId(id)}
              >
                <Icon size={14} aria-hidden="true" />
                {label}
              </button>
            ))}
          </nav>
        </header>
        <main id="main-content">
          <ActiveView navigate={setActiveId} />
        </main>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Add temporary mode components so TypeScript can compile**

Run:

```bash
mkdir -p explorer/src/{explorer-kit,lib,guide,components-deepdive,parallelism,architecture}
```

Create `explorer/src/explorer-kit/mode.ts`:

```ts
import type { ComponentType } from 'react'
import type { LucideIcon } from 'lucide-react'

export interface ExplorerModeProps {
  navigate: (id: string) => void
}

export interface ExplorerMode {
  id: string
  label: string
  icon: LucideIcon
  subtitle: string
  View: ComponentType<ExplorerModeProps>
}
```

Create temporary components:

```bash
for path in \
  explorer/src/guide/GuideExplorer.tsx \
  explorer/src/components-deepdive/ComponentExplorer.tsx \
  explorer/src/parallelism/ParallelismExplorer.tsx \
  explorer/src/architecture/ArchitectureExplorer.tsx
do
  cat > "$path" <<'EOF'
import type { ExplorerModeProps } from '@/explorer-kit/mode'

export default function PlaceholderExplorer(_props: ExplorerModeProps) {
  return <div className="panel p-6 text-sm text-ink-soft">Manifest-backed view loading soon.</div>
}
EOF
done
```

These temporary files will be replaced in later tasks.

- [ ] **Step 5: Add Megatron asset helpers**

Create `explorer/src/lib/assets.ts`:

```ts
export const REPO_HOME = 'https://github.com/NVIDIA/Megatron-LM'
export const DEFAULT_SOURCE_REF = 'main'

export function dataUrl(path: string): string {
  const clean = path.replace(/^\//, '')
  return `${import.meta.env.BASE_URL}${clean}`
}

export function logoMarkUrl(): string {
  return dataUrl('logo-mark.svg')
}

export function sourceUrl(fileLine: string, ref = DEFAULT_SOURCE_REF): string {
  const [file, line] = fileLine.split(':')
  const anchor = line ? `#L${line}` : ''
  return `${REPO_HOME}/blob/${ref}/${file}${anchor}`
}
```

Create `explorer/src/lib/fetch.ts`:

```ts
import { dataUrl } from './assets'

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export async function fetchExplorerJson<T>(path: string): Promise<T> {
  const url = dataUrl(path.startsWith('data/') ? path : `data/${path}`)
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export async function fetchExplorerText(path: string): Promise<string> {
  const url = dataUrl(path.startsWith('data/') ? path : `data/${path}`)
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status} ${response.statusText}`)
  }
  return response.text()
}
```

Create `explorer/src/lib/utils.ts`:

```ts
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

- [ ] **Step 6: Install and typecheck**

Run:

```bash
cd explorer
npm install
npm run typecheck
```

Expected: `tsc -b` exits with status 0.

- [ ] **Step 7: Commit scaffold**

Run:

```bash
git add explorer
git commit -m "feat: scaffold megatron explorer"
```

---

### Task 2: Add Shared Explorer Kit And Workflow Skeleton

**Files:**
- Create/modify: `explorer/src/explorer-kit/AsyncBoundary.tsx`
- Create/modify: `explorer/src/explorer-kit/DetailDrawer.tsx`
- Create/modify: `explorer/src/explorer-kit/SubjectSwitcher.tsx`
- Create/modify: `explorer/src/explorer-kit/ViewTabs.tsx`
- Create: `explorer/scripts/_refs.py`
- Create: `explorer/scripts/workflow.sh`
- Create: `explorer/scripts/verify_manifests.py`

- [ ] **Step 1: Copy shared explorer-kit components from vLLM**

Run:

```bash
cp /Users/phi9t/CodeBase/vllm/explorer/src/explorer-kit/AsyncBoundary.tsx explorer/src/explorer-kit/AsyncBoundary.tsx
cp /Users/phi9t/CodeBase/vllm/explorer/src/explorer-kit/DetailDrawer.tsx explorer/src/explorer-kit/DetailDrawer.tsx
cp /Users/phi9t/CodeBase/vllm/explorer/src/explorer-kit/SubjectSwitcher.tsx explorer/src/explorer-kit/SubjectSwitcher.tsx
cp /Users/phi9t/CodeBase/vllm/explorer/src/explorer-kit/ViewTabs.tsx explorer/src/explorer-kit/ViewTabs.tsx
```

Then replace any literal `vLLM` references with `Megatron` if present.

- [ ] **Step 2: Add source-reference resolver**

Create `explorer/scripts/_refs.py`:

```python
from __future__ import annotations

from pathlib import Path


def resolve_line(repo_root: Path, file: str, symbol: str | None, fallback: int | None = None) -> int | None:
    path = repo_root / file
    if not path.is_file() or not symbol:
        return fallback
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return fallback
    candidates = [symbol]
    parts = symbol.split()
    if len(parts) >= 2:
        candidates.append(f"{parts[0]} {parts[1]}")
    for candidate in candidates:
        for line_no, line in enumerate(lines, start=1):
            idx = line.find(candidate)
            if idx < 0:
                continue
            after = line[idx + len(candidate) : idx + len(candidate) + 1]
            if after and (after.isalnum() or after == "_"):
                continue
            return line_no
    return fallback


def ref(repo_root: Path, file: str, symbol: str | None, fallback: int | None = None) -> str:
    line = resolve_line(repo_root, file, symbol, fallback)
    return f"{file}:{line}" if line else file
```

- [ ] **Step 3: Add manifest verifier**

Create `explorer/scripts/verify_manifests.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "explorer" / "public" / "data"
MAX_LABEL = 28


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_ref(ref: str) -> str | None:
    file = ref.split(":", 1)[0]
    if not (ROOT / file).is_file():
        return f"missing source file: {ref}"
    return None


def check_graph(path: Path) -> list[str]:
    errors: list[str] = []
    data = load(path)
    for node in data.get("nodes", []):
        label = node.get("label", "")
        if len(label) > MAX_LABEL:
            errors.append(f"{path}: label too long: {label}")
        node_ref = node.get("ref") or node.get("file")
        if node_ref:
            err = check_ref(str(node_ref))
            if err:
                errors.append(f"{path}: {err}")
    return errors


def check_model(path: Path) -> list[str]:
    errors: list[str] = []
    data = load(path)
    blocks = list(data.get("prelude", [])) + list(data.get("head", []))
    for group in data.get("layers", []):
        for branch in group.get("branches", []):
            blocks.append(branch.get("preNorm", {}))
            blocks.extend(branch.get("steps", []))
    for block in blocks:
        label = block.get("label", "")
        if len(label) > 28:
            errors.append(f"{path}: block label too long: {label}")
        block_ref = block.get("ref")
        if block_ref:
            err = check_ref(str(block_ref))
            if err:
                errors.append(f"{path}: {err}")
    return errors


def main() -> None:
    errors: list[str] = []
    for graph in (DATA / "graphs").glob("*.json"):
        if graph.name != "index.json":
            errors.extend(check_graph(graph))
    for model in (DATA / "models").glob("*.json"):
        if model.name != "index.json":
            errors.extend(check_model(model))
    if not (DATA / "guide.md").is_file():
        errors.append("missing guide.md")
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("manifest verification passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add workflow script**

Create `explorer/scripts/workflow.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPLORER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPLORER_DIR}/.." && pwd)"
DATA_DIR="${EXPLORER_DIR}/public/data"
PYTHON="${PYTHON:-python3}"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

check_node() {
  command -v node >/dev/null 2>&1 || die "node not found"
  command -v npm >/dev/null 2>&1 || die "npm not found"
}

gen_data() {
  mkdir -p "${DATA_DIR}"
  log "Generating guide"
  "${PYTHON}" "${SCRIPT_DIR}/build_guide.py" --repo-root "${REPO_ROOT}" --out "${DATA_DIR}/guide.md"
  log "Generating component graphs"
  "${PYTHON}" "${SCRIPT_DIR}/build_component_manifest.py" --repo-root "${REPO_ROOT}" --out-dir "${DATA_DIR}/graphs"
  log "Generating parallelism manifests"
  "${PYTHON}" "${SCRIPT_DIR}/build_parallelism_manifest.py" --repo-root "${REPO_ROOT}" --out-dir "${DATA_DIR}/parallelism"
  log "Generating model architecture manifests"
  "${PYTHON}" "${SCRIPT_DIR}/build_model_arch.py" --repo-root "${REPO_ROOT}" --out-dir "${DATA_DIR}/models"
  log "Verifying manifests"
  "${PYTHON}" "${SCRIPT_DIR}/verify_manifests.py"
}

install_deps() {
  check_node
  (cd "${EXPLORER_DIR}" && npm install)
}

build_app() {
  check_node
  (cd "${EXPLORER_DIR}" && npm run build)
}

dev_server() {
  check_node
  (cd "${EXPLORER_DIR}" && npm run dev)
}

verify() {
  gen_data
  build_app
}

case "${1:-all}" in
  all) gen_data; install_deps; build_app ;;
  gen-data) gen_data ;;
  install) install_deps ;;
  build) build_app ;;
  dev) dev_server ;;
  verify) verify ;;
  *) die "unknown phase '$1' (use: all | gen-data | install | build | dev | verify)" ;;
esac
```

Run:

```bash
chmod +x explorer/scripts/workflow.sh
```

- [ ] **Step 5: Typecheck**

Run:

```bash
cd explorer
npm run typecheck
```

Expected: typecheck passes.

- [ ] **Step 6: Commit shared kit and workflow skeleton**

Run:

```bash
git add explorer/src/explorer-kit explorer/scripts
git commit -m "feat: add explorer shared kit and workflow"
```

---

### Task 3: Generate And Render The Guide Mode

**Files:**
- Create: `explorer/scripts/build_guide.py`
- Create/modify: `explorer/public/data/guide.md`
- Replace: `explorer/src/guide/GuideExplorer.tsx`

- [ ] **Step 1: Add guide generator**

Create `explorer/scripts/build_guide.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


SECTIONS = [
    ("README.md", "Megatron-LM and Megatron Core"),
    ("docs/get-started/overview.md", "Project Architecture"),
    ("docs/user-guide/parallelism-guide.md", "Parallelism Strategies"),
    ("docs/ultra-scale-playbook/megatron-mapping.md", "Ultra-Scale Playbook Mapping"),
    ("docs/user-guide/features/moe.md", "Mixture of Experts"),
    ("docs/user-guide/features/multi_latent_attention.md", "Multi-Latent Attention"),
    ("docs/user-guide/features/megatron_fsdp.md", "Megatron FSDP"),
    ("docs/user-guide/features/context_parallel.md", "Context Parallelism"),
]


def excerpt(text: str, max_lines: int = 90) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
        if line.startswith("<!---"):
            continue
        if len(kept) >= max_lines and not in_code:
            break
        kept.append(line)
    return "\n".join(kept).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="explorer/public/data/guide.md")
    args = parser.parse_args()
    repo_root = Path(args.repo_root)
    chunks = [
        "# Megatron Explorer Guide",
        "",
        "This generated guide links the visual explorer back to Megatron-LM source, docs, and example configs.",
        "",
        "```mermaid",
        "flowchart LR",
        "  YAML[Example YAML] --> Launcher[Launcher]",
        "  Launcher --> Train[Training loop]",
        "  Train --> Core[Megatron Core]",
        "  Core --> Parallel[Parallelism groups]",
        "  Core --> Model[Transformer model]",
        "  Train --> Optim[Optimizer and checkpointing]",
        "```",
    ]
    for rel, title in SECTIONS:
        path = repo_root / rel
        if not path.is_file():
            chunks.extend([f"## {title}", "", f"Source `{rel}` was not found.", ""])
            continue
        chunks.extend([f"## {title}", "", f"_Source: `{rel}`_", "", excerpt(path.read_text(encoding="utf-8")), ""])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Replace GuideExplorer**

Create `explorer/src/guide/GuideExplorer.tsx`:

```tsx
import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { errorMessage, fetchExplorerText } from '@/lib/fetch'
import type { ExplorerModeProps } from '@/explorer-kit/mode'

export default function GuideExplorer(_props: ExplorerModeProps) {
  const [markdown, setMarkdown] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchExplorerText('guide.md')
      .then(setMarkdown)
      .catch((err) => setError(errorMessage(err)))
  }, [])

  if (!markdown) {
    return (
      <AsyncBoundary
        loading={error === null}
        error={error}
        loadingLabel="Loading guide..."
        errorPrefix="Failed to load guide.md. Run ./scripts/workflow.sh gen-data"
      />
    )
  }

  return (
    <article className="panel guide-panel p-6">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
    </article>
  )
}
```

- [ ] **Step 3: Generate guide data**

Run:

```bash
python3 explorer/scripts/build_guide.py --repo-root . --out explorer/public/data/guide.md
test -s explorer/public/data/guide.md
```

Expected: `guide.md` exists and is non-empty.

- [ ] **Step 4: Typecheck**

Run:

```bash
cd explorer
npm run typecheck
```

Expected: typecheck passes.

- [ ] **Step 5: Commit Guide mode**

Run:

```bash
git add explorer/scripts/build_guide.py explorer/public/data/guide.md explorer/src/guide/GuideExplorer.tsx
git commit -m "feat: add megatron guide explorer"
```

---

### Task 4: Generate Component Graph Manifests

**Files:**
- Create: `explorer/scripts/build_component_manifest.py`
- Create: `explorer/public/data/graphs/index.json`
- Create: `explorer/public/data/graphs/training-loop.json`
- Create: `explorer/public/data/graphs/core-stack.json`
- Create: `explorer/src/components-deepdive/types.ts`

- [ ] **Step 1: Define component graph types**

Create `explorer/src/components-deepdive/types.ts`:

```ts
export interface ComponentNode {
  id: string
  label: string
  group: 'entry' | 'training' | 'core' | 'parallel' | 'state'
  ref: string
  symbol: string | null
  desc: string
  docs?: string[]
}

export interface ComponentEdge {
  from: string
  to: string
  label?: string
}

export interface ComponentManifest {
  slug: string
  label: string
  generated_at: string
  warnings: string[]
  nodes: ComponentNode[]
  edges: ComponentEdge[]
}

export interface GraphIndexEntry {
  slug: string
  label: string
  manifest: string
}
```

- [ ] **Step 2: Add component manifest generator**

Create `explorer/scripts/build_component_manifest.py` with static node definitions and resolved refs:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _refs import ref  # noqa: E402


def node(repo: Path, id: str, label: str, group: str, file: str, symbol: str | None, desc: str, docs: list[str] | None = None):
    return {
        "id": id,
        "label": label,
        "group": group,
        "ref": ref(repo, file, symbol),
        "symbol": symbol,
        "desc": desc,
        "docs": docs or [],
    }


def training_loop(repo: Path):
    nodes = [
        node(repo, "yaml", "YAML config", "entry", "examples/llama/llama_config.py", "class LlamaConfig", "Pydantic-style config objects turn YAML presets into Megatron CLI arguments.", ["examples/llama/README.md"]),
        node(repo, "launcher", "Launcher", "entry", "examples/llama/run_llama.py", "def main", "The example launcher validates config, prepares runtime paths, and invokes the training command."),
        node(repo, "args", "Arguments", "training", "megatron/training/yaml_arguments.py", "def core_transformer_config_from_yaml", "Megatron normalizes CLI/YAML values into runtime arguments and transformer config."),
        node(repo, "initialize", "Initialize", "training", "megatron/training/initialize.py", "def initialize_megatron", "Distributed initialization, tokenizer setup, random seeds, and model-parallel groups."),
        node(repo, "train", "Training loop", "training", "megatron/training/training.py", "def pretrain", "Top-level pretraining loop wiring model, data iterators, forward/backward, optimizer, evaluation, and checkpointing."),
        node(repo, "dataset", "Datasets", "state", "megatron/core/datasets/gpt_dataset.py", "class GPTDataset", "GPT-style indexed datasets provide token samples to the training loop."),
        node(repo, "model", "Model provider", "core", "megatron/core/models/gpt/gpt_model.py", "class GPTModel", "The GPT model wraps embeddings, transformer layers, final norm, and output projection."),
        node(repo, "schedule", "Forward/backward", "parallel", "megatron/core/pipeline_parallel/schedules.py", "def get_forward_backward_func", "Pipeline schedules drive microbatch forward/backward execution."),
        node(repo, "optimizer", "Optimizer", "state", "megatron/core/optimizer/optimizer.py", "class MegatronOptimizer", "Optimizer wrapper coordinates parameter updates, grad scaling, clipping, and distributed state."),
        node(repo, "checkpoint", "Checkpointing", "state", "megatron/training/checkpointing.py", "def save_checkpoint", "Checkpoint save/load preserves model, optimizer, RNG, and distributed state."),
    ]
    edges = [
        ("yaml", "launcher"), ("launcher", "args"), ("args", "initialize"), ("initialize", "train"),
        ("train", "dataset"), ("train", "model"), ("train", "schedule"), ("schedule", "optimizer"),
        ("optimizer", "checkpoint"), ("checkpoint", "train"),
    ]
    return manifest("training-loop", "Training Loop", nodes, edges)


def core_stack(repo: Path):
    nodes = [
        node(repo, "gpt", "GPTModel", "core", "megatron/core/models/gpt/gpt_model.py", "class GPTModel", "GPTModel assembles embeddings, transformer block, final norm, and output layer."),
        node(repo, "block", "TransformerBlock", "core", "megatron/core/transformer/transformer_block.py", "class TransformerBlock", "A stack of transformer layers built from module specs."),
        node(repo, "layer", "TransformerLayer", "core", "megatron/core/transformer/transformer_layer.py", "class TransformerLayer", "Self-attention, residual paths, layer norms, dense MLP, or MoE."),
        node(repo, "attention", "Attention", "core", "megatron/core/transformer/attention.py", "class Attention", "Attention projection, core attention, and output projection."),
        node(repo, "mlp", "Dense MLP", "core", "megatron/core/transformer/mlp.py", "class MLP", "Dense feed-forward block used by standard transformer layers."),
        node(repo, "moe", "MoELayer", "core", "megatron/core/transformer/moe/moe_layer.py", "class MoELayer", "Mixture-of-Experts routing, dispatch, expert compute, and combine."),
        node(repo, "tp", "Tensor parallel", "parallel", "megatron/core/tensor_parallel/layers.py", "class ColumnParallelLinear", "Layer-level sharding for large projections."),
        node(repo, "pp", "Pipeline parallel", "parallel", "megatron/core/pipeline_parallel/schedules.py", "def get_forward_backward_func", "Layer-depth partitioning and microbatch schedules."),
        node(repo, "ddp", "Distributed", "parallel", "megatron/core/distributed/distributed_data_parallel.py", "class DistributedDataParallel", "Data-parallel gradient synchronization and overlap."),
        node(repo, "optim", "Optimizer", "state", "megatron/core/optimizer/distrib_optimizer.py", "class DistributedOptimizer", "Distributed optimizer state and parameter shard handling."),
    ]
    edges = [
        ("gpt", "block"), ("block", "layer"), ("layer", "attention"), ("layer", "mlp"),
        ("layer", "moe"), ("attention", "tp"), ("mlp", "tp"), ("moe", "tp"),
        ("block", "pp"), ("gpt", "ddp"), ("ddp", "optim"),
    ]
    return manifest("core-stack", "Megatron Core Stack", nodes, edges)


def manifest(slug: str, label: str, nodes: list[dict], edges: list[tuple[str, str]]):
    return {
        "slug": slug,
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for a, b in edges],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="explorer/public/data/graphs")
    args = parser.parse_args()
    repo = Path(args.repo_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = [training_loop(repo), core_stack(repo)]
    for graph in graphs:
        (out / f"{graph['slug']}.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    index = [{"slug": g["slug"], "label": g["label"], "manifest": f"graphs/{g['slug']}.json"} for g in graphs]
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate graph data**

Run:

```bash
python3 explorer/scripts/build_component_manifest.py --repo-root . --out-dir explorer/public/data/graphs
python3 -m json.tool explorer/public/data/graphs/index.json >/dev/null
python3 -m json.tool explorer/public/data/graphs/training-loop.json >/dev/null
python3 -m json.tool explorer/public/data/graphs/core-stack.json >/dev/null
```

Expected: all JSON files parse.

- [ ] **Step 4: Commit component data**

Run:

```bash
git add explorer/scripts/build_component_manifest.py explorer/public/data/graphs explorer/src/components-deepdive/types.ts
git commit -m "feat: generate megatron component graphs"
```

---

### Task 5: Render Component Graphs

**Files:**
- Create/replace: `explorer/src/components-deepdive/ArchitectureGraph.tsx`
- Create/replace: `explorer/src/components-deepdive/ComponentDrawer.tsx`
- Replace: `explorer/src/components-deepdive/ComponentExplorer.tsx`

- [ ] **Step 1: Implement SVG graph renderer**

Create `explorer/src/components-deepdive/ArchitectureGraph.tsx`:

```tsx
import { KIND_COLOR } from '@/architecture/blockTypes'
import type { ComponentEdge, ComponentNode } from './types'

interface Props {
  nodes: ComponentNode[]
  edges: ComponentEdge[]
  selectedId: string | null
  onSelect: (node: ComponentNode) => void
}

const COL_W = 220
const ROW_H = 86
const CARD_W = 170
const CARD_H = 52

const GROUP_COL: Record<ComponentNode['group'], number> = {
  entry: 0,
  training: 1,
  core: 2,
  parallel: 3,
  state: 4,
}

const GROUP_COLOR: Record<ComponentNode['group'], string> = {
  entry: KIND_COLOR.embed,
  training: KIND_COLOR.proj,
  core: KIND_COLOR.attn,
  parallel: KIND_COLOR.rope,
  state: KIND_COLOR.mlp,
}

function layout(nodes: ComponentNode[]) {
  const counts = new Map<number, number>()
  return new Map(
    nodes.map((node) => {
      const col = GROUP_COL[node.group]
      const row = counts.get(col) ?? 0
      counts.set(col, row + 1)
      return [node.id, { x: 30 + col * COL_W, y: 30 + row * ROW_H }]
    }),
  )
}

export default function ArchitectureGraph({ nodes, edges, selectedId, onSelect }: Props) {
  const positions = layout(nodes)
  const width = 30 + 5 * COL_W
  const height = Math.max(280, 60 + Math.max(...Array.from(positions.values()).map((p) => p.y)) + CARD_H)
  const byId = new Map(nodes.map((node) => [node.id, node]))

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="diagram-svg" role="group" aria-label="Component graph">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(148,163,184,0.75)" />
        </marker>
      </defs>
      {edges.map((edge) => {
        const a = positions.get(edge.from)
        const b = positions.get(edge.to)
        if (!a || !b) return null
        const x1 = a.x + CARD_W
        const y1 = a.y + CARD_H / 2
        const x2 = b.x
        const y2 = b.y + CARD_H / 2
        const mid = (x1 + x2) / 2
        return (
          <path
            key={`${edge.from}-${edge.to}`}
            d={`M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`}
            fill="none"
            stroke="rgba(148,163,184,0.55)"
            strokeWidth={1.5}
            markerEnd="url(#arrow)"
          />
        )
      })}
      {nodes.map((node) => {
        const p = positions.get(node.id)!
        const color = GROUP_COLOR[node.group]
        const selected = selectedId === node.id
        return (
          <g
            key={node.id}
            role="button"
            tabIndex={0}
            aria-label={`${node.label}: ${node.desc}`}
            transform={`translate(${p.x} ${p.y})`}
            onClick={() => onSelect(node)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onSelect(node)
              }
            }}
            className="diagram-node"
          >
            <rect width={CARD_W} height={CARD_H} rx={10} fill="rgba(15,23,42,0.88)" stroke={selected ? '#38bdf8' : color} strokeWidth={selected ? 2.5 : 1.2} />
            <rect width={4} height={CARD_H - 14} x={8} y={7} rx={2} fill={color} />
            <text x={22} y={23} fill="#f3f4f6" fontSize={12} fontWeight={700}>{node.label}</text>
            <text x={22} y={40} fill="#9ca3af" fontSize={9}>{node.group}</text>
          </g>
        )
      })}
      {nodes.length === 0 && <text x={20} y={40} fill="#9ca3af">No component nodes in manifest.</text>}
      {Array.from(byId.keys()).length === 0 ? null : null}
    </svg>
  )
}
```

- [ ] **Step 2: Implement component drawer**

Create `explorer/src/components-deepdive/ComponentDrawer.tsx`:

```tsx
import { ExternalLink } from 'lucide-react'
import { sourceUrl } from '@/lib/assets'
import type { ComponentNode } from './types'

interface Props {
  node: ComponentNode | null
  onNavigate: (id: string) => void
}

export default function ComponentDrawer({ node, onNavigate }: Props) {
  if (!node) {
    return (
      <aside className="drawer panel p-5">
        <p className="text-sm text-ink-soft">Select a component to inspect its role and source location.</p>
      </aside>
    )
  }

  return (
    <aside className="drawer panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-xs uppercase text-cyan">{node.group}</p>
          <h2 className="mt-1 text-xl font-semibold text-ink">{node.label}</h2>
        </div>
        <a href={sourceUrl(node.ref)} target="_blank" rel="noopener noreferrer" className="icon-link" aria-label="Open source">
          <ExternalLink size={16} />
        </a>
      </div>
      <p className="mt-4 text-sm leading-6 text-ink-soft">{node.desc}</p>
      <dl className="mt-4 space-y-2 text-sm">
        <div>
          <dt className="font-mono text-xs text-ink-muted">Source</dt>
          <dd className="font-mono text-xs text-cyan">{node.ref}</dd>
        </div>
        {node.symbol && (
          <div>
            <dt className="font-mono text-xs text-ink-muted">Symbol</dt>
            <dd className="font-mono text-xs text-ink-soft">{node.symbol}</dd>
          </div>
        )}
      </dl>
      {node.docs && node.docs.length > 0 && (
        <div className="mt-4">
          <p className="font-mono text-xs uppercase text-ink-muted">Related docs</p>
          <ul className="mt-2 space-y-1">
            {node.docs.map((doc) => (
              <li key={doc} className="font-mono text-xs text-ink-soft">{doc}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="mt-5 flex flex-wrap gap-2">
        <button type="button" className="family-switch-btn" onClick={() => onNavigate('parallelism')}>Parallelism</button>
        <button type="button" className="family-switch-btn" onClick={() => onNavigate('architecture')}>Architecture</button>
      </div>
    </aside>
  )
}
```

- [ ] **Step 3: Implement component explorer**

Create `explorer/src/components-deepdive/ComponentExplorer.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { SubjectSwitcher } from '@/explorer-kit/SubjectSwitcher'
import { errorMessage, fetchExplorerJson } from '@/lib/fetch'
import type { ExplorerModeProps } from '@/explorer-kit/mode'
import ArchitectureGraph from './ArchitectureGraph'
import ComponentDrawer from './ComponentDrawer'
import type { ComponentManifest, ComponentNode, GraphIndexEntry } from './types'

export default function ComponentExplorer({ navigate }: ExplorerModeProps) {
  const [index, setIndex] = useState<GraphIndexEntry[] | null>(null)
  const [slug, setSlug] = useState<string>('')
  const [manifest, setManifest] = useState<ComponentManifest | null>(null)
  const [selected, setSelected] = useState<ComponentNode | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchExplorerJson<GraphIndexEntry[]>('graphs/index.json')
      .then((entries) => {
        setIndex(entries)
        setSlug(entries[0]?.slug ?? '')
      })
      .catch((err) => setError(errorMessage(err)))
  }, [])

  useEffect(() => {
    if (!index || !slug) return
    const entry = index.find((item) => item.slug === slug)
    if (!entry) return
    setManifest(null)
    setSelected(null)
    setError(null)
    fetchExplorerJson<ComponentManifest>(entry.manifest)
      .then(setManifest)
      .catch((err) => setError(errorMessage(err)))
  }, [index, slug])

  if (!index || !manifest) {
    return <AsyncBoundary loading={error === null} error={error} loadingLabel="Loading component graph..." errorPrefix="Failed to load component graph. Run ./scripts/workflow.sh gen-data" />
  }

  return (
    <div className="flex flex-col gap-5">
      <SubjectSwitcher
        label="Subsystem map"
        ariaLabel="Subsystem map"
        value={slug}
        options={index.map((entry) => ({ value: entry.slug, label: entry.label }))}
        onChange={setSlug}
      />
      <div className="grid items-start gap-5 lg:grid-cols-[1fr_380px]">
        <section className="panel p-5">
          <div className="mb-4">
            <h2 className="text-xl font-semibold text-ink">{manifest.label}</h2>
            <p className="text-sm text-ink-soft">Click a node for source and context.</p>
          </div>
          <ArchitectureGraph nodes={manifest.nodes} edges={manifest.edges} selectedId={selected?.id ?? null} onSelect={setSelected} />
        </section>
        <div className="lg:sticky lg:top-6">
          <ComponentDrawer node={selected} onNavigate={navigate} />
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Typecheck and build**

Run:

```bash
cd explorer
npm run typecheck
npm run build
```

Expected: both pass.

- [ ] **Step 5: Commit component renderer**

Run:

```bash
git add explorer/src/components-deepdive
git commit -m "feat: render megatron component graphs"
```

---

### Task 6: Generate Parallelism Manifest

**Files:**
- Create: `explorer/scripts/build_parallelism_manifest.py`
- Create: `explorer/public/data/parallelism/index.json`
- Create: `explorer/public/data/parallelism/strategies.json`
- Create: `explorer/src/parallelism/types.ts`

- [ ] **Step 1: Define parallelism types**

Create `explorer/src/parallelism/types.ts`:

```ts
export interface ParallelStrategy {
  id: 'dp' | 'tp' | 'pp' | 'cp' | 'ep' | 'fsdp'
  label: string
  objective: string
  bestFor: string
  flags: string[]
  sourceRefs: string[]
  desc: string
}

export interface ParallelPreset {
  id: string
  label: string
  worldSize: number
  tp: number
  pp: number
  cp: number
  ep: number
  note: string
}

export interface ParallelismManifest {
  generated_at: string
  strategies: ParallelStrategy[]
  presets: ParallelPreset[]
}
```

- [ ] **Step 2: Add parallelism generator**

Create `explorer/scripts/build_parallelism_manifest.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


STRATEGIES = [
    {
        "id": "dp",
        "label": "Data Parallelism",
        "objective": "Batch dimension",
        "bestFor": "Data scalability and standard training",
        "flags": ["--data-parallel-sharding-strategy"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "megatron/core/distributed/distributed_data_parallel.py"],
        "desc": "Replicate or shard model state across ranks while splitting the batch.",
    },
    {
        "id": "tp",
        "label": "Tensor Parallelism",
        "objective": "Individual layers",
        "bestFor": "Large hidden dimensions and projection matrices",
        "flags": ["--tensor-model-parallel-size", "--sequence-parallel"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "megatron/core/tensor_parallel/layers.py"],
        "desc": "Shard large linear layers and attention heads across ranks.",
    },
    {
        "id": "pp",
        "label": "Pipeline Parallelism",
        "objective": "Model depth",
        "bestFor": "Very deep models",
        "flags": ["--pipeline-model-parallel-size", "--num-layers-per-virtual-pipeline-stage"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "megatron/core/pipeline_parallel/schedules.py"],
        "desc": "Partition layers by depth and schedule microbatches through stages.",
    },
    {
        "id": "cp",
        "label": "Context Parallelism",
        "objective": "Sequence length",
        "bestFor": "Long sequence training",
        "flags": ["--context-parallel-size", "--cp-comm-type"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "docs/user-guide/features/context_parallel.md"],
        "desc": "Split sequence/context work across ranks to reduce activation pressure.",
    },
    {
        "id": "ep",
        "label": "Expert Parallelism",
        "objective": "MoE experts",
        "bestFor": "Mixture-of-Experts models",
        "flags": ["--expert-model-parallel-size", "--num-experts", "--moe-grouped-gemm"],
        "sourceRefs": ["docs/user-guide/parallelism-guide.md", "megatron/core/transformer/moe/moe_layer.py"],
        "desc": "Distribute expert weights and route tokens to top-k experts.",
    },
    {
        "id": "fsdp",
        "label": "Megatron FSDP",
        "objective": "Model state",
        "bestFor": "Large models with data-parallel state pressure",
        "flags": ["--use-megatron-fsdp", "--data-parallel-sharding-strategy"],
        "sourceRefs": ["docs/user-guide/features/megatron_fsdp.md", "megatron/core/distributed/fsdp/mcore_fsdp_adapter.py"],
        "desc": "Shard parameters, gradients, and optimizer state across data-parallel ranks.",
    },
]


PRESETS = [
    {"id": "llama3-8b", "label": "Llama3 8B local", "worldSize": 8, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "note": "Matches examples/llama/configs/llama3_8b_long.yaml."},
    {"id": "llama70b", "label": "Llama 70B guide", "worldSize": 64, "tp": 4, "pp": 4, "cp": 2, "ep": 1, "note": "From the parallelism guide recommendation table."},
    {"id": "deepseek-v3", "label": "DeepSeek-V3 guide", "worldSize": 1024, "tp": 2, "pp": 16, "cp": 1, "ep": 64, "note": "MoE example from the parallelism guide."},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="explorer/public/data/parallelism")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "strategies": STRATEGIES, "presets": PRESETS}
    (out / "strategies.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "index.json").write_text(json.dumps([{"slug": "strategies", "label": "Parallelism Strategies", "manifest": "parallelism/strategies.json"}], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate parallelism data**

Run:

```bash
python3 explorer/scripts/build_parallelism_manifest.py --repo-root . --out-dir explorer/public/data/parallelism
python3 -m json.tool explorer/public/data/parallelism/strategies.json >/dev/null
```

Expected: JSON parses.

- [ ] **Step 4: Commit parallelism data**

Run:

```bash
git add explorer/scripts/build_parallelism_manifest.py explorer/public/data/parallelism explorer/src/parallelism/types.ts
git commit -m "feat: generate parallelism manifest"
```

---

### Task 7: Render Parallelism Mode

**Files:**
- Create: `explorer/src/parallelism/RankGrid.tsx`
- Create: `explorer/src/parallelism/StrategyDiagram.tsx`
- Replace: `explorer/src/parallelism/ParallelismExplorer.tsx`

- [ ] **Step 1: Add rank-grid component**

Create `explorer/src/parallelism/RankGrid.tsx`:

```tsx
interface Props {
  worldSize: number
  tp: number
  pp: number
  cp: number
  ep: number
}

export function inferredDp(worldSize: number, tp: number, pp: number, cp: number, ep: number): number | null {
  const denom = tp * pp * cp * ep
  if (denom <= 0 || worldSize % denom !== 0) return null
  return worldSize / denom
}

export default function RankGrid({ worldSize, tp, pp, cp, ep }: Props) {
  const dp = inferredDp(worldSize, tp, pp, cp, ep)
  const ranks = Array.from({ length: Math.min(worldSize, 128) }, (_, rank) => rank)
  return (
    <div className="panel p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2 font-mono text-xs text-ink-soft">
        <span>world={worldSize}</span>
        <span>TP={tp}</span>
        <span>PP={pp}</span>
        <span>CP={cp}</span>
        <span>EP={ep}</span>
        <span className={dp === null ? 'text-danger' : 'text-success'}>DP={dp ?? 'invalid'}</span>
      </div>
      <div className="rank-grid">
        {ranks.map((rank) => (
          <div key={rank} className="rank-cell" title={`rank ${rank}`}>
            {rank}
          </div>
        ))}
      </div>
      {worldSize > 128 && <p className="mt-2 text-xs text-ink-muted">Showing first 128 ranks.</p>}
      {dp === null && (
        <p className="mt-3 text-sm text-danger">
          Invalid: world_size must be divisible by TP * PP * CP * EP = {tp * pp * cp * ep}.
        </p>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Add strategy cards**

Create `explorer/src/parallelism/StrategyDiagram.tsx`:

```tsx
import { sourceUrl } from '@/lib/assets'
import type { ParallelStrategy } from './types'

interface Props {
  strategies: ParallelStrategy[]
  selectedId: string
  onSelect: (id: string) => void
}

export default function StrategyDiagram({ strategies, selectedId, onSelect }: Props) {
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {strategies.map((strategy) => (
        <button
          key={strategy.id}
          type="button"
          className={`panel p-4 text-left transition ${selectedId === strategy.id ? 'border-panelborder-active bg-panel-hover' : ''}`}
          onClick={() => onSelect(strategy.id)}
        >
          <p className="font-mono text-xs uppercase text-cyan">{strategy.id}</p>
          <h3 className="mt-1 text-base font-semibold text-ink">{strategy.label}</h3>
          <p className="mt-2 text-sm text-ink-soft">{strategy.desc}</p>
          <div className="mt-3 flex flex-wrap gap-1">
            {strategy.flags.map((flag) => (
              <span key={flag} className="rounded bg-black/25 px-2 py-1 font-mono text-[11px] text-ink-soft">{flag}</span>
            ))}
          </div>
          <div className="mt-3 space-y-1">
            {strategy.sourceRefs.map((ref) => (
              <a key={ref} href={sourceUrl(ref)} target="_blank" rel="noopener noreferrer" className="block font-mono text-[11px] text-cyan" onClick={(event) => event.stopPropagation()}>
                {ref}
              </a>
            ))}
          </div>
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Implement ParallelismExplorer**

Create `explorer/src/parallelism/ParallelismExplorer.tsx`:

```tsx
import { useEffect, useMemo, useState } from 'react'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { ViewTabs } from '@/explorer-kit/ViewTabs'
import { errorMessage, fetchExplorerJson } from '@/lib/fetch'
import type { ExplorerModeProps } from '@/explorer-kit/mode'
import RankGrid from './RankGrid'
import StrategyDiagram from './StrategyDiagram'
import type { ParallelismManifest } from './types'

export default function ParallelismExplorer(_props: ExplorerModeProps) {
  const [manifest, setManifest] = useState<ParallelismManifest | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState('tp')
  const [preset, setPreset] = useState('llama3-8b')
  const [worldSize, setWorldSize] = useState(8)
  const [tp, setTp] = useState(1)
  const [pp, setPp] = useState(1)
  const [cp, setCp] = useState(1)
  const [ep, setEp] = useState(1)

  useEffect(() => {
    fetchExplorerJson<ParallelismManifest>('parallelism/strategies.json')
      .then((data) => {
        setManifest(data)
        const first = data.presets[0]
        if (first) {
          setPreset(first.id)
          setWorldSize(first.worldSize)
          setTp(first.tp)
          setPp(first.pp)
          setCp(first.cp)
          setEp(first.ep)
        }
      })
      .catch((err) => setError(errorMessage(err)))
  }, [])

  const presetOptions = useMemo(
    () => manifest?.presets.map((item) => ({ value: item.id, label: item.label })) ?? [],
    [manifest],
  )

  if (!manifest) {
    return <AsyncBoundary loading={error === null} error={error} loadingLabel="Loading parallelism data..." errorPrefix="Failed to load parallelism manifest. Run ./scripts/workflow.sh gen-data" />
  }

  function applyPreset(id: string) {
    const next = manifest?.presets.find((item) => item.id === id)
    if (!next) return
    setPreset(id)
    setWorldSize(next.worldSize)
    setTp(next.tp)
    setPp(next.pp)
    setCp(next.cp)
    setEp(next.ep)
  }

  return (
    <div className="flex flex-col gap-5">
      <ViewTabs ariaLabel="Parallelism preset" value={preset} onChange={applyPreset} options={presetOptions} />
      <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
        <section className="panel p-5">
          <h2 className="text-lg font-semibold text-ink">Rank controls</h2>
          {[
            ['World', worldSize, setWorldSize],
            ['TP', tp, setTp],
            ['PP', pp, setPp],
            ['CP', cp, setCp],
            ['EP', ep, setEp],
          ].map(([label, value, setter]) => (
            <label key={label as string} className="mt-4 block text-sm text-ink-soft">
              <span className="mb-1 block font-mono text-xs">{label as string}: {value as number}</span>
              <input className="w-full" type="range" min={1} max={label === 'World' ? 1024 : 64} value={value as number} onChange={(event) => (setter as (n: number) => void)(Number(event.target.value))} />
            </label>
          ))}
        </section>
        <RankGrid worldSize={worldSize} tp={tp} pp={pp} cp={cp} ep={ep} />
      </div>
      <StrategyDiagram strategies={manifest.strategies} selectedId={selected} onSelect={setSelected} />
    </div>
  )
}
```

- [ ] **Step 4: Add CSS for rank grid**

Append to `explorer/src/index.css`:

```css
.rank-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(42px, 1fr));
  gap: 6px;
}

.rank-cell {
  min-height: 34px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.72);
  color: #f3f4f6;
  font-family: var(--font-mono);
  font-size: 11px;
  display: grid;
  place-items: center;
}
```

- [ ] **Step 5: Typecheck and build**

Run:

```bash
cd explorer
npm run typecheck
npm run build
```

Expected: both pass.

- [ ] **Step 6: Commit parallelism renderer**

Run:

```bash
git add explorer/src/parallelism explorer/src/index.css
git commit -m "feat: render parallelism explorer"
```

---

### Task 8: Generate Model Architecture Manifests

**Files:**
- Create: `explorer/src/architecture/modelArch.ts`
- Create: `explorer/src/architecture/blockTypes.ts`
- Create: `explorer/scripts/build_model_arch.py`
- Create: `explorer/public/data/models/*.json`

- [ ] **Step 1: Copy architecture type/math baseline from vLLM**

Run:

```bash
cp /Users/phi9t/CodeBase/vllm/explorer/src/architecture/modelArch.ts explorer/src/architecture/modelArch.ts
cp /Users/phi9t/CodeBase/vllm/explorer/src/architecture/blockTypes.ts explorer/src/architecture/blockTypes.ts
```

Then adjust comments only if they name vLLM-specific internals. Keep the model math and lens formatter.

- [ ] **Step 2: Add Megatron model architecture generator**

Create `explorer/scripts/build_model_arch.py` by adapting `/Users/phi9t/CodeBase/vllm/explorer/scripts/build_model_arch.py` with these required changes:

```python
MODEL_LIST = [
    ("qwen3-0_6b", "Qwen3-0.6B", "dense-qknorm"),
    ("qwen3-8b", "Qwen3-8B", "dense-qknorm"),
    ("qwen3-30b-a3b", "Qwen3-30B-A3B", "moe-qknorm"),
    ("deepseek-v3", "DeepSeek-V3", "mla-moe"),
]
```

Use these Megatron source files for block refs:

```python
MEGATRON_REFS = {
    "embed": ("megatron/core/models/gpt/gpt_model.py", "self.embedding"),
    "transformer": ("megatron/core/transformer/transformer_block.py", "class TransformerBlock"),
    "layer": ("megatron/core/transformer/transformer_layer.py", "class TransformerLayer"),
    "attention": ("megatron/core/transformer/attention.py", "class Attention"),
    "mlp": ("megatron/core/transformer/mlp.py", "class MLP"),
    "moe": ("megatron/core/transformer/moe/moe_layer.py", "class MoELayer"),
    "router": ("megatron/core/transformer/moe/router.py", "class Router"),
    "shared": ("megatron/core/transformer/moe/shared_experts.py", "class SharedExpertMLP"),
    "mla": ("megatron/core/transformer/multi_latent_attention.py", "class MultiLatentAttention"),
    "head": ("megatron/core/models/gpt/gpt_model.py", "self.output_layer"),
}
```

Keep the vLLM fallback config values for Qwen3 and DeepSeek-V3. The generated block `symbol` strings should name Megatron concepts, for example `GPTModel.embedding`, `TransformerLayer.self_attention`, `MoELayer`, and `MultiLatentAttention`.

- [ ] **Step 3: Generate model data**

Run:

```bash
python3 explorer/scripts/build_model_arch.py --repo-root . --out-dir explorer/public/data/models
python3 -m json.tool explorer/public/data/models/index.json >/dev/null
python3 -m json.tool explorer/public/data/models/qwen3-0_6b.json >/dev/null
python3 -m json.tool explorer/public/data/models/qwen3-8b.json >/dev/null
python3 -m json.tool explorer/public/data/models/qwen3-30b-a3b.json >/dev/null
python3 -m json.tool explorer/public/data/models/deepseek-v3.json >/dev/null
```

Expected: all JSON files parse.

- [ ] **Step 4: Run manifest verification**

Run:

```bash
python3 explorer/scripts/verify_manifests.py
```

Expected: `manifest verification passed`.

- [ ] **Step 5: Commit model manifests**

Run:

```bash
git add explorer/scripts/build_model_arch.py explorer/src/architecture/modelArch.ts explorer/src/architecture/blockTypes.ts explorer/public/data/models
git commit -m "feat: generate megatron model architecture manifests"
```

---

### Task 9: Render Model Architecture Mode

**Files:**
- Copy/modify: `explorer/src/architecture/ModelCircuit.tsx`
- Replace: `explorer/src/architecture/ArchitectureExplorer.tsx`

- [ ] **Step 1: Copy the circuit renderer**

Run:

```bash
cp /Users/phi9t/CodeBase/vllm/explorer/src/architecture/ModelCircuit.tsx explorer/src/architecture/ModelCircuit.tsx
cp /Users/phi9t/CodeBase/vllm/explorer/src/architecture/ArchitectureExplorer.tsx explorer/src/architecture/ArchitectureExplorer.tsx
```

- [ ] **Step 2: Replace vLLM copy in ArchitectureExplorer**

In `explorer/src/architecture/ArchitectureExplorer.tsx`, ensure these strings are Megatron-specific:

```tsx
const emptyMessage = 'Run ./scripts/workflow.sh gen-data to generate model architecture manifests.'
```

Drawer/source copy should refer to Megatron Core source, not vLLM model executor source.

- [ ] **Step 3: Typecheck and build**

Run:

```bash
cd explorer
npm run typecheck
npm run build
```

Expected: both pass.

- [ ] **Step 4: Commit architecture renderer**

Run:

```bash
git add explorer/src/architecture
git commit -m "feat: render megatron model architecture explorer"
```

---

### Task 10: Wire Full Data Workflow And README

**Files:**
- Modify: `explorer/scripts/workflow.sh`
- Create/modify: `explorer/README.md`
- Modify: `explorer/src/lib/assets.ts`

- [ ] **Step 1: Verify workflow runs all generators**

Run:

```bash
./explorer/scripts/workflow.sh gen-data
```

Expected:

```text
manifest verification passed
```

- [ ] **Step 2: Update README**

Create `explorer/README.md`:

````markdown
# Megatron Explorer

An interactive, static explorer for Megatron-LM and Megatron Core internals.

## Modes

- Guide: generated inner-workings guide.
- Components: source-linked training loop and Megatron Core graphs.
- Parallelism: TP, PP, DP, CP, EP, and FSDP rank geometry.
- Model Architecture: Qwen3 and DeepSeek-V3 circuits grounded in Megatron Core source.

## Run

```bash
cd explorer
./scripts/workflow.sh gen-data
./scripts/workflow.sh install
./scripts/workflow.sh dev
```

## Verify

```bash
cd explorer
./scripts/workflow.sh verify
```

The explorer is a documentation and visualization tool. It does not launch Megatron training jobs.
````

- [ ] **Step 3: Confirm source URL branch**

In `explorer/src/lib/assets.ts`, keep:

```ts
export const DEFAULT_SOURCE_REF = 'main'
```

If the active development branch must be used for local Pages deployment later, change only this constant.

- [ ] **Step 4: Commit docs and workflow**

Run:

```bash
git add explorer/README.md explorer/scripts/workflow.sh explorer/src/lib/assets.ts
git commit -m "docs: document megatron explorer workflow"
```

---

### Task 11: Final Verification And Local Server

**Files:**
- No planned file changes unless verification reveals issues.

- [ ] **Step 1: Run full verification**

Run:

```bash
cd explorer
./scripts/workflow.sh verify
```

Expected:

```text
manifest verification passed
```

and `npm run build` exits with status 0.

- [ ] **Step 2: Run lint**

Run:

```bash
cd explorer
npm run lint
```

Expected: ESLint exits with status 0.

- [ ] **Step 3: Start dev server**

Run:

```bash
cd explorer
npm run dev -- --host 127.0.0.1
```

Expected: Vite prints a local URL such as `http://127.0.0.1:5173/`.

- [ ] **Step 4: Browser smoke checks**

Open the Vite URL and verify:

- Guide renders Markdown.
- Components mode shows both subject graphs and drawer source links.
- Parallelism mode changes rank geometry and marks invalid tuples.
- Architecture mode switches across Qwen3-0.6B, Qwen3-8B, Qwen3-30B-A3B, and DeepSeek-V3.
- Tab navigation reaches family switcher buttons, subject switchers, graph nodes, and circuit nodes.
- No text overlaps at mobile width and desktop width.

- [ ] **Step 5: Commit any verification fixes**

If files changed during verification, run:

```bash
git add explorer
git commit -m "fix: polish megatron explorer verification"
```

If no files changed, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage:
  - Standalone `megatronlm/explorer`: Task 1.
  - Four modes: Tasks 3, 5, 7, 9.
  - Static manifest generators: Tasks 3, 4, 6, 8.
  - Qwen3 and DeepSeek-V3: Tasks 8 and 9.
  - Source-linked drawers: Tasks 4 and 5.
  - Parallelism controls and invalid tuples: Tasks 6 and 7.
  - Workflow and verification: Tasks 2, 10, 11.
- Placeholder scan: this plan avoids forbidden placeholder markers and unnamed implementation steps.
- Type consistency:
  - `ExplorerModeProps` is defined in Task 1 and used by every mode.
  - Component graph types are defined before graph rendering.
  - Parallelism types are defined before parallelism rendering.
  - Architecture types/math are copied before architecture rendering.
