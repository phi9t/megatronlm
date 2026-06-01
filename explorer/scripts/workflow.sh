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
  for generator in \
    build_guide.py \
    build_component_manifest.py \
    build_parallelism_manifest.py \
    build_model_arch.py
  do
    generator_path="${SCRIPT_DIR}/${generator}"
    [[ -f "${generator_path}" ]] || die "generator not implemented yet: ${generator_path}"
  done

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
