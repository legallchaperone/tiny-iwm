# scirpts

`scripts` folder contains bash scripts for you to scale up your project on cloud.
Don't put your jupyter notebooks here! They belongs to `debug` folder.

General scripts that are useful for all projects can be put in the `script` folder directly.

---

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.

## M1 codec gate

`python -m scripts.validate_m1_codec` runs a selected validation or test split
through the strict SANA reader and a frozen codec supplied by a local
`module.path:factory`. It writes one atomic JSON report and exits nonzero if any
sample fails reading, alignment, encode/decode, shape, or causality checks. See
`python -m scripts.validate_m1_codec --help` for all required paths and options.
