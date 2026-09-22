"""Pure readiness decision logic for the CWX-22 Stage A gate."""

from __future__ import annotations

import math
from typing import Any


def evaluate_readiness(report: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "formal_length_executable": report["formal_length"]["completed"],
        "traceable_profile": all(
            report["traceability"].get(key)
            for key in ("profile_artifact_sha256", "training_artifact_sha256", "checkpoint_id")
        ),
        "checkpoint_resume_verified": report["checkpoint_resume"]["verified"],
        "condition_preserved": report["short_term_generation"]["condition_max_abs_error"] == 0.0,
        "generation_finite": all(
            math.isfinite(float(report["short_term_generation"][key]))
            for key in ("target_mse", "output_mean", "output_std")
        ),
        "camera_response_measurable": (
            math.isfinite(float(report["camera_conditioning"]["counterfactual_mean_abs_delta"]))
            and report["camera_conditioning"]["counterfactual_mean_abs_delta"] > 0
        ),
    }
    mechanical_ready = all(checks.values())
    return {
        "checks": checks,
        "mechanically_ready_for_stage_b_initialization": mechanical_ready,
        "status": "passed_with_quality_warning" if mechanical_ready else "failed",
    }
