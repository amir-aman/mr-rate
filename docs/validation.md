# Local validation record

The current dataset scope is batch26. Live download details and the prepared validation
subset are recorded in [batch26.md](batch26.md). No pretrained model benchmark is claimed.

Validated November 13, 2026 in `D:\Projects\MR-Rate\code` on Windows with Python
3.13.13. The local `.venv` was created with `--system-site-packages` to reuse existing
numerical/PyTorch libraries; project additions were installed in that environment.
The README recommends an isolated environment for new installations. Exact direct
dependency versions are recorded in [requirements-tested.txt](requirements-tested.txt);
that file is an environment record, not a complete portable transitive lock.

## Executed successfully

- Editable install of the package, including `.[dev,model]` dependency resolution.
- `python -m pytest -q`: **61 passed**, all with outbound socket connections blocked.
  The final run set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid an unrelated globally
  installed pytest plugin warning. Plain pytest also executed the tests; disabling
  plugin auto-loading is optional for a clean new environment.
- `python -m ruff check .`: all checks passed.
- `python -m ruff format --check src tests scripts`: all 24 Python files formatted.
- `python -m mrrate_vlm demo-synthetic`: complete 12-study fixture → manifest → NIfTI
  slices → fake embeddings → retrieval → bootstrap/duplicates/subgroups → saved outputs.
- Separate CLI evaluation of that saved synthetic embedding bundle, with 12 candidates.
- PowerShell pilot dry-run via
  `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_pilot.ps1 -MaxStudies 2 -DryRun`.
  Local script execution is otherwise disabled by this host's PowerShell policy.
- CLI help and metadata/image/model dry-run tests.
- Visual inspection of synthetic metric bars and a three-slice QC montage; both are
  explicitly labeled SYNTHETIC.
- All ten extracted handoff documents were byte-compared with the original ZIP and
  remained identical.

## Coverage of the handoff acceptance criteria

| Requirement | Evidence |
|---|---|
| Series/study joins do not multiply samples | `test_dataset.py`: many-series fixture, identical/conflicting duplicates |
| ZIP batch comes from metadata filename | Fixture deliberately has metadata batch01 and patient-first split batch00 |
| Official patient split integrity | Leakage, missing membership, and patient-disagreement tests |
| One study per patient | Repeated-patient selection and manifest validation tests |
| Dry-runs fetch no data or model | Socket-blocked CLI/download tests, forbidden model-loader stub |
| NIfTI orientation, finite data, RGB | LAS→RAS synthetic volumes, NaN/Inf replacement, shape/channel assertions |
| Empty/long report handling | Empty/fallback tests, tokenizer-based counts and actual 64-token API test |
| Exact ranking/pooling/controls | Hand-calculated matrices, nontrivial ranks, tie handling, derangement |
| CPU operation without weights | Full synthetic pipeline and tiny randomly initialized SigLIP API test |
| Private artifacts ignored | Git's own ignore matcher tested against data, reports, tokens and checkpoints |
| Download limits and cache | HEAD-size guard, selected-file fetches, validated ZIP reuse, cache-corruption recovery |
| Evaluation provenance | Changed manifest rejected; reordered rows realigned; duplicate sensitivity checked |
| Batch26 scope | Metadata filtered before reading other batch CSVs; manifest scope enforced at inference/evaluation |
| Explicit full-batch download | One MRI folder only, all support files, combined size cap and checksum-failure state |

The API test creates a tiny random `SiglipModel`, an actual local fast tokenizer and
official image processor. It exercises `MedSiglipEncoder` on CPU using Transformers
5.23.0, with no pretrained weights. It checks frozen/eval state, independent tower
outputs, batching, dimensions, normalization, output-object adaptation and truncation.
If the optional model dependencies are absent, this single test skips; core tests and
the synthetic demo remain available without installing PyTorch or Transformers.

## Live data verification and remaining limits

Live gated access was used to download and verify all **311 batch26 MRI ZIPs and 63
supporting files**, totaling **8,931,357,493 bytes**, in the ignored local data directory.
`data/raw/batch26_download_summary.json` records `status: complete`, all 374 verified file
entries, and the resolved repository revision. No other MRI batch was downloaded.
Real batch26 CSV ingestion, global patient split integrity, and the 12-patient validation
manifest passed. The local `qc` command processed all 12 selected NIfTIs successfully:
RAS orientation, three unique slices per study, zero nonfinite voxels, and 12 private
QC montages. This is preprocessing verification; anatomical/clinical manual review is
still required before interpreting any experiment.
The MedSigLIP checkpoint has **not** been downloaded and no real-data retrieval performance
has been measured. The actual pretrained tokenizer/checkpoint, GPU inference/mixed
precision, real model runtime and resource use remain unverified. The Linux shell runner
and GitHub Actions matrix are supplied but were not
executed on this Windows host. The template CI installs the lightweight core/dev extras;
the optional actual-model API test will skip there unless model dependencies are added.

The synthetic encoder reconstructs phantoms from known text pattern IDs by design.
Its perfect matched retrieval tests implementation alignment and cannot support a
scientific performance claim. All generated demo outputs remain under ignored
`data/synthetic/`; the public README contains no real benchmark numbers.

## Next model experiment

Both gates have been accepted for the current account. Install the optional model extras
and authenticate privately with `hf auth login` when intentionally starting model inference.
The session-only data-download token is not saved in project files. Use `config/batch26.yaml`,
inspect `selection_summary.json`, `brain_screening_audit.csv`, and private QC montages. Use
official validation for pipeline decisions and a separate test configuration for the
final held-out run. Preserve the resolved dataset/model commit SHAs from provenance.
Real outputs are written by `evaluate` only after an actual embedding run; no synthetic
metrics are copied into the real results directory.
