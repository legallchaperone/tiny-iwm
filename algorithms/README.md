# algorithms

`algorithms` folder is designed to contain implementation of algorithms or models.
Content in `algorithms` can be loosely grouped components (e.g. models) or an algorithm has already has all
components chained together (e.g. Lightning Module, RL algo).
You should create a folder name after your own algorithm or baselines in it.

Two example can be found in `examples` subfolder.

The `common` subfolder is designed to contain general purpose classes that's useful for many projects, e.g MLP.

You should not run any `.py` file from algorithms folder.
Instead, you write unit tests / debug python files in `debug` and launch script in `experiments`.

You are discouraged from putting visualization utilities in algorithms, as those should go to `utils` in project root.

## World model

`algorithms/world_model/models/JointVideoDiT` is the only DiT implementation
used by Stage A, Stage B, and inference. It accepts codec latents in
`[B, C, T, H, W]`, patches all three video axes, runs joint attention over the
resulting tokens, and returns a velocity tensor with the original latent shape.

The model has no pretrained-weight loading API. Its depth, width, head count,
patch size, latent channels, and MLP ratio come from configuration. The default
24-layer, width-832 configuration is approximately 303M trainable parameters;
tests instantiate a much smaller configuration with the same code path.

Bidirectional and causal execution differ only in the boolean
`visibility_mask` passed to `forward` (`True` means the query may see the key).
`time_offset` keeps later rollout chunks on the episode's global timeline.
Callers may request names from `model.observation_points` through `capture`;
returned activations retain gradients for later regularizers and probes.

Each algorithm class takes in a DictConfig file `cfg` in its `__init__`, which allows you to pass in arguments via configuration file in `configurations/algorithm` or [command line override](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/).

---

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.
