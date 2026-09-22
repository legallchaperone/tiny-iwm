The `datasets` folder is used to contain dataset code or environment code.
Don't store actual data like images here! For those, please use the `data` folder instead of `datasets`.

`datasets/sana_wm` implements the strict M1 adapter. Its JSON manifest is the
source of truth for sample identity, source scene, split, and data version.
Paths are relative to `dataset.data_root`. Missing or malformed video, camera,
or metadata files raise `SampleReadError`; training code should record that
failure and exclude the sample rather than inventing camera values. Image
resize/crop must call the paired transforms so RGB-pixel intrinsics stay aligned.

Run the complete M1 gate with `python -m scripts.validate_m1_codec`. It accepts a
local manifest/data root and a zero-argument codec factory in
`package.module:callable` form. The JSON report includes exact first/last layout
mappings, camera timestamp alignment, held-out reconstruction and latent
statistics, a future-frame causality perturbation, the canonical cache key, and
a structured failure list. A passing report requires every selected validation
or test sample to pass; it never substitutes synthetic camera data or a fallback
codec.

Create a folder to create your own pytorch dataset definition. Then, update the `__init__.py`
at every level to register all datasets.

Each dataset class takes in a DictConfig file `cfg` in its `__init__`, which allows you to pass in arguments via configuration file in `configurations/dataset` or [command line override](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).

---

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.
