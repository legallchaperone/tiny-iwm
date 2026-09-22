from scripts.stage_a_readiness import evaluate_readiness


def test_readiness_requires_resume_condition_camera_and_traceability() -> None:
    report = {
        "formal_length": {"completed": True},
        "traceability": {
            "profile_artifact_sha256": "sha256:a",
            "training_artifact_sha256": "sha256:b",
            "checkpoint_id": "sha256:c",
        },
        "checkpoint_resume": {"verified": True},
        "short_term_generation": {
            "condition_max_abs_error": 0.0,
            "target_mse": 2.0,
            "output_mean": 0.0,
            "output_std": 1.0,
        },
        "camera_conditioning": {"counterfactual_mean_abs_delta": 0.01},
    }
    decision = evaluate_readiness(report)
    assert decision["mechanically_ready_for_stage_b_initialization"] is True
    assert decision["status"] == "passed_with_quality_warning"

    report["camera_conditioning"]["counterfactual_mean_abs_delta"] = 0.0
    assert evaluate_readiness(report)["status"] == "failed"
