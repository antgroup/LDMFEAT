# FEAT-38M checkpoint

The model file is distributed separately under [Apache-2.0](../LICENSE) and is not tracked in Git. The intended distribution location is [GitHub Releases](https://github.com/antgroup/LDMFEAT/releases); a public weight download has not yet been verified for this source snapshot. Consult the release description for availability.

Place the downloaded `FEAT-38M.pt` in this directory. All inference entry points use this filename by default; `--model_path` accepts another local path.

Expected size: **155,418,367 bytes**.

Expected SHA-256:

```text
84acc3161c52ce67f99a43d202008b5c9f25826aff2d5b917cbfe6ced8ba6ade
```

From the repository root, verify the file with `sha256sum checkpoints/FEAT-38M.pt` on Linux or `shasum -a 256 checkpoints/FEAT-38M.pt` on macOS. The size and hash must match before loading.
