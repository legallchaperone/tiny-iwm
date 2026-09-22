from omegaconf import OmegaConf

from algorithms.examples.helloworld.example_algos import ExampleAlgo, ExampleBackwardAlgo


def test_example_algo_wraps_message():
    cfg = OmegaConf.create({"debug": False, "prefix": "[", "suffix": "]"})

    assert ExampleAlgo(cfg).run("Hello") == "[Hello]"


def test_example_backward_algo_reverses_and_wraps_message():
    cfg = OmegaConf.create({"debug": False, "prefix": "(", "suffix": ")"})

    assert ExampleBackwardAlgo(cfg).run("Hello") == "(olleH)"
