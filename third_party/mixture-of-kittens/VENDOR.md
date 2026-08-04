# Vendored: mixture-of-kittens

| Field | Value |
| --- | --- |
| Upstream | https://github.com/cursor/mixture-of-kittens |
| Upstream HEAD (vendored) | `8f90b740693b60832e09eef7ce0d33c389c44b86` |
| Upstream message | Initial public release |
| Upstream date | 2026-08-03 |
| License | Apache-2.0 (`LICENSE`) |
| Nested dependency | `third_party/ThunderKittens` from https://github.com/HazyResearch/ThunderKittens at `1c3920d993404dd49a6d4c7267ea11d583bd5c68` (as recorded by upstream `.gitmodules` pin) |

## Why it is here

Vendored for local study and integration with Megatron-LM MoE paths without relying
on a live network clone at build/read time. MoK requires Blackwell SM100/SM103,
CUDA 13+, PyTorch 2.10+ — see upstream `README.md`.

## Refresh procedure

From repository root (run on a machine with git network access):

```bash
TMP=local/mixture-of-kittens-src
rm -rf "$TMP"
git clone --depth 1 --recurse-submodules https://github.com/cursor/mixture-of-kittens.git "$TMP"
rsync -a --delete --exclude '.git' --exclude '**/.git' --exclude '**/.github' "$TMP/" third_party/mixture-of-kittens/
# update SHAs in this file to match `git -C $TMP rev-parse HEAD` and the ThunderKittens pin
```

Do not edit vendored sources in place for Megatron product changes; instead
document adapter notes outside this tree (or upstream PRs). Nested
`third_party/ThunderKittens` is expanded as a normal directory (not a git submodule
of this repository) so a single clone of Megatron-LM is self-contained.

## Install (optional)

From `third_party/mixture-of-kittens` with a matching CUDA/PyTorch stack:

```bash
cd third_party/mixture-of-kittens
pip install . --no-build-isolation
# or SM100: MOK_ARCH=SM100 pip install . --no-build-isolation
```
