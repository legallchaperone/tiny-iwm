# configurations

We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. Change/Add the yaml files in this folder
to change the default configurations. You can also override the default configurations by
passing command line arguments.

The default `config.yaml` composes the M0 world-model skeleton and performs no
training (`experiment.tasks=[]`). Scientific values that still require an
experiment or a memory profile are written as `null`, rather than receiving an
implicit guess. W&B is disabled by default, so composing and launching this
baseline locally requires no credentials. Example configurations remain
available through explicit Hydra overrides.

All configurations are automatically saved in wandb run.

---

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.
