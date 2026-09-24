"""Resolve and construct the model shared by training, rollout, and probes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from algorithms.world_model.models import JointVideoDiT, JointVideoDiTConfig
from algorithms.world_model.flow import FlowMatchSpec
from algorithms.world_model.objectives import CosineDFObjective, CosineDFSchedule, FMObjective
from algorithms.world_model.training_policies import TrainingPolicy, build_training_policy
from inference.samplers import DFDDIMSampler, FMEulerSampler


_COMPONENT_FIELDS = {
    "block": "block_kind",
    "attention": "attention_kind",
    "ffn": "ffn_kind",
    "norm": "norm_kind",
    "position": "position_kind",
    "conditioner": "conditioner_kind",
    "latent_io": "latent_io_kind",
}
_MODEL_NAMES = {
    "joint_spatiotemporal_dit": "gelu_tanh",
    "joint_spatiotemporal_dit_swiglu": "swiglu",
}
_LEGACY_STAGE_B_RUN = "stage-b-sekai-subset-seed21-v2"


@dataclass(frozen=True)
class ModelCapabilities:
    camera_projection: bool
    visibility_mask: bool = True
    kv_cache: bool = True
    feature_capture: bool = True


@dataclass(frozen=True)
class BuiltModel:
    model: JointVideoDiT
    architecture: str
    capabilities: ModelCapabilities


@dataclass(frozen=True)
class BuiltRecipe:
    model: BuiltModel
    objective: FMObjective | CosineDFObjective
    policy: TrainingPolicy
    sampler: FMEulerSampler | DFDDIMSampler
    sampling_steps: int

    def __post_init__(self) -> None:
        if self.objective.name != self.sampler.objective_name:
            raise ValueError("objective and sampler names are incompatible")
        if self.objective.prediction_type != self.sampler.prediction_type:
            raise ValueError("objective prediction type differs from sampler")


def resolve_model_config(root: Mapping[str, Any]) -> tuple[str, JointVideoDiTConfig]:
    """Resolve named model/components without silently choosing defaults for new fields."""
    if "model" not in root:
        raise ValueError("model configuration is required")
    model_values = root["model"]
    if not isinstance(model_values, Mapping):
        raise ValueError("model configuration must be a mapping")
    architecture = model_values.get("architecture")
    if architecture is None and root.get("run_id") == _LEGACY_STAGE_B_RUN:
        architecture = "joint_spatiotemporal_dit"
    if architecture not in _MODEL_NAMES:
        raise ValueError(
            f"unsupported model architecture {architecture!r}; choose {sorted(_MODEL_NAMES)}"
        )
    if model_values.get("initialization", "random") != "random":
        raise ValueError("model initialization must be random; checkpoint loading is separate")
    components = model_values.get("components", {})
    if not isinstance(components, Mapping):
        raise ValueError("model.components must be a mapping")
    unknown_components = set(components) - set(_COMPONENT_FIELDS)
    if unknown_components:
        raise ValueError(f"unsupported model components: {sorted(unknown_components)}")
    field_names = {item.name for item in fields(JointVideoDiTConfig)}
    allowed = field_names | {"architecture", "initialization", "target_parameters", "components", "attention"}
    unknown = set(model_values) - allowed
    if unknown:
        raise ValueError(f"unsupported model fields: {sorted(unknown)}")
    values = {key: model_values[key] for key in field_names if key in model_values}
    if "attention" in model_values:
        if "attention_kind" in values and values["attention_kind"] != model_values["attention"]:
            raise ValueError("model.attention conflicts with model.attention_kind")
        values["attention_kind"] = model_values["attention"]
    for component, field in _COMPONENT_FIELDS.items():
        if component in components:
            if field in values and values[field] != components[component]:
                raise ValueError(f"model.{field} conflicts with model.components.{component}")
            values[field] = components[component]
    expected_ffn = _MODEL_NAMES[architecture]
    if "ffn_kind" in values and values["ffn_kind"] != expected_ffn:
        raise ValueError(
            f"{architecture} requires model.components.ffn={expected_ffn}"
        )
    values["ffn_kind"] = expected_ffn
    try:
        values["patch_size"] = tuple(values["patch_size"])
        config = JointVideoDiTConfig(**values)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid model configuration: {exc}") from exc
    conditioning = root.get("conditioning")
    if isinstance(conditioning, Mapping):
        camera = conditioning.get("camera")
        if isinstance(camera, Mapping):
            if camera.get("enabled") is False and config.prope_camera_dims is not None:
                raise ValueError(
                    "conditioning.camera.enabled=false requires model.prope_camera_dims=null"
                )
            if camera.get("enabled") is True and config.prope_camera_dims is None:
                raise ValueError(
                    "conditioning.camera.enabled=true requires model.prope_camera_dims"
                )
    return architecture, config


def build_model(root: Mapping[str, Any]) -> BuiltModel:
    """Build a selected implementation; never load weights implicitly."""
    architecture, config = resolve_model_config(root)
    model = JointVideoDiT(config)
    return BuiltModel(
        model=model,
        architecture=architecture,
        capabilities=ModelCapabilities(camera_projection=config.prope_camera_dims is not None),
    )


def resolve_recipe_selection(root: Mapping[str, Any]) -> tuple[str, str, str, int]:
    """Resolve objective/policy/sampler and reject mismatched old stage fields."""
    if all(key in root for key in ("objective", "training_policy", "sampler")):
        for key in ("objective", "training_policy", "sampler"):
            if not isinstance(root[key], Mapping):
                raise ValueError(f"{key} configuration must be a mapping")
        objective_name = root["objective"].get("name")
        policy_name = root["training_policy"].get("name")
        sampler_name = root["sampler"].get("name")
        steps = root["sampler"].get("steps")
        stage = root.get("stage")
        if isinstance(stage, Mapping):
            expected_policy = {
                "bidirectional": "bidirectional",
                "causal_tf": "teacher_forced_causal",
                "causal_direct": "teacher_forced_causal",
            }.get(stage.get("name"))
            if expected_policy is not None and policy_name != expected_policy:
                raise ValueError(
                    f"stage {stage.get('name')} conflicts with training policy {policy_name}"
                )
    elif root.get("run_id") in {
        "stage-a-sekai-subset-seed21-v1", "stage-b-sekai-subset-seed21-v2"
    }:
        objective_name = "native_fm"
        policy_name = "bidirectional" if root["stage"] == "A" else "teacher_forced_causal"
        sampler_name, steps = "fm_euler", 16
    else:
        raise ValueError("recipe requires objective, training_policy, and sampler groups")
    if type(steps) is not int or steps <= 0:
        raise ValueError("sampler.steps must be a positive integer")
    compatible = {
        "native_fm": ("fm_euler", {"bidirectional", "teacher_forced_causal"}),
        "minimal_df": ("df_ddim", {"teacher_forced_causal"}),
    }
    if objective_name not in compatible:
        raise ValueError(f"unsupported objective: {objective_name}")
    required_sampler, policies = compatible[objective_name]
    if sampler_name != required_sampler:
        raise ValueError(
            f"objective {objective_name} requires sampler {required_sampler}, got {sampler_name}"
        )
    if policy_name not in policies:
        raise ValueError(f"objective {objective_name} cannot use policy {policy_name}")
    prediction = {"native_fm": "velocity", "minimal_df": "epsilon"}[objective_name]
    if "objective" in root and root["objective"].get("prediction_type") != prediction:
        raise ValueError(f"objective {objective_name} requires prediction_type={prediction}")
    if "sampler" in root and root["sampler"].get("prediction_type") != prediction:
        raise ValueError(f"sampler {sampler_name} requires prediction_type={prediction}")
    return objective_name, policy_name, sampler_name, steps


def build_recipe(root: Mapping[str, Any]) -> BuiltRecipe:
    """Construct one compatible model/objective/policy/sampler recipe."""
    objective_name, policy_name, _, steps = resolve_recipe_selection(root)
    model = build_model(root)
    policy = build_training_policy(policy_name)
    if objective_name == "native_fm":
        values = root.get("objective", {}).get("flow", {})
        spec = FlowMatchSpec(**values)
        objective = FMObjective(spec, policy)
        sampler = FMEulerSampler()
    else:
        values = root.get("objective", {}).get("schedule", {})
        schedule = CosineDFSchedule(**values)
        objective = CosineDFObjective(schedule, policy)
        sampler = DFDDIMSampler(schedule)
        if steps > schedule.train_steps:
            raise ValueError("DF sampler.steps exceeds objective schedule.train_steps")
    return BuiltRecipe(model, objective, policy, sampler, steps)
