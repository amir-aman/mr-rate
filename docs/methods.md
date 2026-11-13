# Method and audit contract

## Units, joins and selection

`patient_uid` groups participants, `study_uid` links an examination and report, and
`series_id` selects a NIfTI within that examination. All identifiers are strings.
Identical repeated rows are collapsed; conflicting repeated keys fail. Metadata
must have unique `(study_uid, series_id)` keys, reports and splits unique `study_uid`
keys. Each study must have one patient and one metadata source batch. Splits are
checked globally, including studies not selected for the pilot. Mismatched patient
IDs or missing official split membership fail. Validated many-to-one merges prevent
Cartesian expansion. `splits.batch_id` is never used to locate MRI files.

Default screening inspects **only available** configured anatomy fields and the report's
`technique`. The audit records observed columns and the rule responsible for each
series decision. Brain/head/cranial/cranium/cerebral/intracranial keywords are positive;
spine/spinal/cervical/thoracic/lumbar/lumbosacral/sacral keywords veto the series. Any
mixed anatomy description is conservatively rejected. This is an auditable screen,
not a validated anatomical classifier. Absent positive evidence, candidates fail.
It does not infer anatomy from T1w/T2w/FLAIR sequence labels, findings, pathology or
clinical indications. Add a genuine body-part column to `brain_columns` if it becomes
available in the downloaded schema. An explicit opt-out labels the run's screen disabled.

The batch26 metadata retain Turkish DICOM descriptions. Brain evidence also recognizes
`beyin`, `kranial`, and `kraniyal`; spine vetoes also recognize `vertebra`, `vertebral`,
`servikal`, `torakal`, `lomber`, `sakral`, and `lumbosakral`. Available protocol and
procedure-description fields are included, so an explicitly spinal series can be rejected
even when its study description mentions brain anatomy. Both regex patterns are saved in
the selection summary. These remain keyword heuristics requiring local manual QC.

`source_batches` defaults to `[batch26]`. File selection applies this restriction **before
reading metadata and report CSVs**, avoiding unnecessary loading of the entire repository's
supporting tables. The complete official split table is still validated for patient leakage.
The summary lists available and actually loaded metadata batches. Embedding/evaluation
commands reject a manifest that contains studies outside the configured allowed batches.
Use a matching `--source-batches` option or explicit YAML change to work on other batches.
The independent synthetic demo clears the real-data batch restriction for its own fixtures.

Derived scans are rejected; missing/unknown derived flags are not treated as false.
Available localizer/subtraction flags must also be false. Sequence priority is FLAIR,
T2w, T1w, then axial preference, lower numeric SeriesNumber and lexical series ID.
There is no claim that acquisition order proves absence of contrast enhancement.
Report targets are NFKC/whitespace cleaned impressions of at least 10 characters by
default. Optional findings fallback must be explicitly enabled and is marked per row.
Clinical information is unused. One selected study per patient is retained by a
SHA-256 ordering of `[seed, study_uid]`, then the requested official split is capped.

The private manifest records source, selected series, target text, official split,
anatomy evidence, text source, dataset revision and optional scanner/QC fields. The
selection summary records missingness, modality counts and stage-wise attrition.
No series is silently substituted when the selected image is unavailable: the run
fails so that exclusions can be inspected and any changed cohort made explicit.

## Native-space image representation

The exact `study/img/study_series.nii.gz` member is copied into a temporary generated
filename. Other members are not extracted. Absolute/traversing paths, Windows path
separators, drive prefixes, symlinks and duplicate selected members are rejected. Member
size and image voxel-count limits guard memory/disk use. A valid 3D shape, invertible
finite affine, and positive finite spacing are required. NaN/Inf values become zero.

Nibabel canonicalizes to closest RAS axes; there is no registration, isotropic resampling
or de-obliquing. QC records original/canonical orientation, spacing and obliquity. The
nonzero foreground heuristic selects axial indices near 20%, 50% and 80% of eligible
foreground extent. Eligibility requires at least 0.5% of the in-plane area, four voxels,
and 10% of the maximum per-slice foreground count. Slices must be unique and nonblank.
Whole-volume foreground 1st/99th percentiles define one consistent grayscale window.
Channel repetition creates uint8 RGB input; the official processor resizes to the model
resolution. Display is anterior-up, patient's right on screen right. Masks are optional
dataset artifacts and are not used by this baseline.

The frozen vision tower processes slices in bounded batches under inference mode.
Each slice vector is unit normalized, vectors are averaged, and the mean is normalized.
The text tower independently embeds the cleaned target. Image inference never receives
the target text. Fixed-length padding and truncation follow SigLIP conventions. Original
and retained **actual tokenizer** token counts include special tokens, up to 64. The
fake synthetic encoder uses explicitly labeled whitespace counts, not model token counts.

Per-study cache keys include the ZIP's SHA-256, complete input row fingerprint, code
hash, full configuration, resolved model revision, dataset revision and environment.
An atomic consolidated NPZ stores arrays, private study indices, input fingerprints,
provenance and QC. Evaluation aligns by these indices and rejects changed manifests.

## Retrieval and uncertainty

Within one official split, N MRI queries rank exactly N paired reports from distinct
patients. Candidate order is seeded/shuffled, and known target indices are remapped
explicitly. Cosine scores are dot products of normalized vectors. Exact score ties
receive average ranks; this avoids privileging file order when reports are identical.
R@K uses the average-rank threshold, so tied reports may all miss R@1. This conservative
tie convention must accompany comparisons with other implementations.

- R@1 = fraction of associated reports with rank ≤ 1.
- R@5 = fraction with rank ≤ 5; omitted (`null`) when N < 5.
- MRR = mean reciprocal rank; median rank is also reported.
- Uniform random expectation: R@1 = 1/N, R@5 = 5/N when N ≥ 5,
  MRR = harmonic(N)/N, expected rank = (N+1)/2. Random-order assumptions exclude ties.
- The negative control deranges associated labels using a seeded random cycle and
  reranks the same scores. It has no accidentally unchanged matching pairs.
- A percentile 95% bootstrap resamples unique-patient query outcomes 2,000 times,
  conditional on the fixed candidate pool. The intervals are descriptive.
- Duplicate diagnostics casefold and retain alphanumeric tokens. Exact duplicate
  member rate counts all queries in repeated-text groups. Near duplicates use pairwise
  `SequenceMatcher(autojunk=False)` with a default 0.9 threshold, excluding exact pairs.
  This is lexical similarity, not a semantic or clinical equivalence judgment.
- Duplicate-free sensitivity retains the first deterministic representative per
  normalized impression and rebuilds a smaller pool, reporting its own chance baseline.
- Sequence and manufacturer groups use ranks from the full pool, and require at least
  10 query patients by default. Smaller groups report counts and a reason for omission.

The selected batch subset is a convenience pilot. Even large N does not establish
clinical validity, unbiased population sampling or generalization. Freeze decisions
on official validation before using test. No report-derived pathology labels, model
fine-tuning, report generation or Alzheimer's outcomes are implemented.

Generated examples contain only hashed IDs, ranks and scores. They support local
best/worst-case review with private QC, and are excluded from Git along with all other
generated outputs. Hashes are pseudonyms, not a guarantee of anonymity or permission
to redistribute derived data.
