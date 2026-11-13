# Batch26 working dataset

The authorized download selects only `mri/batch26/*.zip` and all repository files
outside `mri/`. No other MRI batch or pretrained model is part of this download.

Pinned repository: `Forithmus/MR-RATE` at
`f6e39794e7820c96c18465231a30da424e6d1c69`.

| Selection | Files | Compressed bytes | Decimal GB |
|---|---:|---:|---:|
| MRI batch26 | 311 ZIPs | 7,417,613,221 | 7.418 |
| All supporting files | 63 | 1,513,744,272 | 1.514 |
| Total | 374 | 8,931,357,493 | 8.931 |

Supporting files include all available metadata/report batch CSVs, pathology labels,
the official split table, dataset README and repository information. The per-file
download/verification state is private in `data/raw/batch26_download_summary.json`;
`status: complete` means every selected file has been downloaded and verified. Local
cache/state files add a little disk overhead beyond the repository file sizes.

**Completed and verified November 13, 2026:** all 374 files are present, all MRI ZIPs
passed CRC validation, and available remote SHA-256 hashes matched. All 12 selected
validation NIfTIs also passed local preprocessing and produced three-slice montages
under `data/qc/batch26-validation/`. The pretrained model has not been downloaded.

The default configuration and `config/batch26.yaml` restrict the experiment to batch26.
Only its metadata/report CSVs are parsed when building a manifest, while `splits.csv`
is checked globally. The extra downloaded metadata do not cause images from other
batches to be selected. The pilot runner now defaults to batch26 as well.

Actual metadata contain 1,172 series, 311 studies, and 307 unique patients. The official
study split counts before filtering are 277 train, 15 validation (`val` in the raw table),
and 19 test. These are study counts, not a claim of 311 independent evaluation patients.
After conservative English/Turkish anatomical screening, valid sequence selection,
nonempty impressions and one study per patient, the current validation manifest contains
**12 unique patients/studies**: 10 FLAIR and 2 T2w selected series. This is a small pilot;
no pretrained retrieval performance has been measured. An uncapped pre-split selection
has 257 unique patients with eligible report targets; the default cap remains 30 per split.

## Use the local download

From the `code` directory, these commands read local metadata or show an offline plan:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m mrrate_vlm --config config/batch26.yaml build-manifest
& $python -m mrrate_vlm --config config/batch26.yaml embed --dry-run
& $python -m mrrate_vlm --config config/batch26.yaml qc
```

`qc` loads only the selected local NIfTIs and writes private slice montages to
`data/qc/batch26-validation/`, with a machine-readable QC summary. It uses no model
or network access and fails clearly if any selected study ZIP is still missing.

Once you intentionally want to download/load the MedSigLIP checkpoint and run inference:

```powershell
& $python -m mrrate_vlm --config config/batch26.yaml embed
& $python -m mrrate_vlm --config config/batch26.yaml evaluate
```

Those latter commands require your privately configured Hugging Face authentication for
the separate model gate. Check the local QC montages, anatomy audit and truncation counts
before interpreting results. The test split remains reserved for a final held-out run.

## Reproduce or resume the selected batch download

With `hf auth login` or a private `HF_TOKEN` environment variable already configured:

```powershell
& $python -m mrrate_vlm --config config/batch26.yaml download-batch --batch 26 --inspect-remote
& $python -m mrrate_vlm --config config/batch26.yaml download-batch --batch 26 --max-total-gb 10
```

The first command requests a remote file-size inventory only. The second downloads or
reuses cached files. An alternative session-only hidden token prompt is available:

```powershell
& $python scripts/download_batch.py --batch batch26 --max-total-gb 10 `
  --revision f6e39794e7820c96c18465231a30da424e6d1c69
```

Tokens are never written to project files or passed as command-line arguments. Raw
images, reports, labels, manifests, checksums, caches and logs stay under ignored `data/`.
