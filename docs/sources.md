# Official implementation references

Reviewed November 13, 2026. These are engineering references; no new empirical model
performance is claimed. The preserved handoff has additional scientific background.

- [MR-RATE native-space dataset and gated terms](https://huggingface.co/datasets/Forithmus/MR-RATE)
- [Official MR-RATE dataset guide](https://github.com/forithmus/MR-RATE/blob/main/data-preprocessing/docs/dataset_guide.md)
- [Official preprocessing and download documentation](https://github.com/forithmus/MR-RATE/blob/main/data-preprocessing/README.md)
- [MedSigLIP model card, access terms and citation](https://huggingface.co/google/medsiglip-448)
- [Google MedSigLIP quickstart](https://github.com/Google-Health/medsiglip/blob/main/notebooks/quick_start_with_hugging_face.ipynb)
- [Hugging Face SigLIP API](https://huggingface.co/docs/transformers/model_doc/siglip)
- [Nibabel orientation utilities](https://nipy.org/nibabel/image_orientation.html)

Implementation decisions verified against the guide: metadata/report/split cardinalities,
`source_batch` from metadata filenames, patient-first meaning of `splits.batch_id`,
native ZIP member layout, current pending body-part labels, sequence labels and optional
DICOM fields. Actual downloaded headers are inspected and recorded; optional columns
are never assumed to exist.

The model card documents AutoModel/AutoProcessor, 448-pixel inputs and a 64-token text
context. The installed Transformers 5.23.0 source was also inspected: feature methods
return a pooling output object in that version. The adapter supports that and the
tensor outputs of Transformers 4. Tests instantiate a tiny randomly initialized
SigLIP on CPU without any downloaded checkpoint; this checks API plumbing only.
