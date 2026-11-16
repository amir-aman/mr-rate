# Explorer source

The page in `docs/` is built from this folder. Run all commands from the repository root.

```bash
pip install nilearn scikit-image trimesh fast-simplification pandas
npm install three esbuild
python explorer/make_assets.py   # brain mesh and slice atlas from the MNI ICBM152 template
python explorer/make_stats.py    # aggregate statistics from local MR-RATE tables
npx esbuild explorer/src/app.js --bundle --minify --format=iife --target=es2020 --outfile=docs/assets/app.js
```

`make_stats.py` reads the gated MR-RATE tables from `data/raw/`, which is ignored by Git. Check the
file paths at the top of the script against your local download. Only `aggregates/stats_public.json` is
meant to be committed. It holds aggregate counts, and cells with fewer than 5 studies are withheld.
Delete `aggregates/stats_full_private.json` after a run.
