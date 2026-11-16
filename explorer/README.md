# Explorer source

`docs/index.html` is built from this folder as one self-contained file. The brain mesh, slice images,
fonts, three.js and the statistics are all embedded, so the page works when opened from disk, in a
preview, and on GitHub Pages.

Rebuild from the repository root:

```bash
npm install three esbuild
python explorer/build_page.py
```

To regenerate the inputs:

```bash
pip install nilearn scikit-image trimesh fast-simplification pandas
python explorer/make_assets.py   # MNI ICBM152 brain mesh and slice atlas into explorer/aggregates/
python explorer/make_stats.py    # aggregate statistics from local MR-RATE tables
```

`make_stats.py` reads the gated MR-RATE tables from `data/raw/`, which is ignored by Git. Check the
file paths at the top of the script against your local download. Only `aggregates/stats_public.json`
is meant to be committed. It holds aggregate counts, and cells with fewer than 5 studies are withheld.
Delete `aggregates/stats_full_private.json` after a run.
