# Perception confidence

Perception confidence is kept separate from visibility (v_t) and action reliability (c_t). A detector's raw score is not a probability. Calibration maps it to the observed probability that a predeclared task success criterion is met.

## Workflow

1. Define entities and success criteria in confidence.entities. For example, an instance mask can be successful when its IoU with reviewed ground truth is at least 0.5. Keep the entity list aligned with the prediction source.
2. Create the episode split and score every split without using labels:

       svdt split -c configs/dataset/cabinet.yaml --backend local
       svdt score -c configs/dataset/cabinet.yaml --backend local --all-episodes --stride 5 -o visibility.segmentation.backend=prompted

   The Cabinet profile leaves segmentation disabled by default. Enable a confidence-producing detector for backend extraction, or supply `--predictions` from an instance-level model. A backend that emits no raw scores produces explicit unavailable observations; it cannot produce calibrated confidence.

3. Prepare reviewed labels at confidence.labels_path. Calibration uses only rows from validation episodes. Test labels are used only for the final calibration report.
4. Calibrate and attach task-level q_perception to the score sidecar:

       svdt confidence-calibrate -c configs/dataset/cabinet.yaml

5. Compare confidence-based weighting with the visibility/reliability baseline:

       svdt weights -c configs/dataset/cabinet.yaml -o weighting.method=perception_only
       svdt train -c configs/train/cabinet_smolvla.yaml -o weighting.method=perception_only
       svdt train -c configs/train/cabinet_smolvla.yaml -o weighting.method=perception_reliability
       svdt train -c configs/train/cabinet_smolvla.yaml -o weighting.method=combined_three_way

Use the same train/validation/test episodes and training budget for every condition. Tune calibration and beta coefficients on validation episodes only.

## CSV input

When predictions come from the configured segmentation backend, the command reads per-role, per-camera raw scores already collected by svdt score. For a detector that emits instance-level records, pass --predictions with:

    key,entity_id,entity_role,output_type,camera_id,timestamp,fps,stride,raw_confidence,raw_score_type,model_id,model_version,condition_id,severity,available,missing_reason

The labels file contains:

    key,entity_id,output_type,camera_id,quality_label,label_source

quality_label is the reviewed result for the configured success criterion. Use 1/true for success and 0/false for failure. Keep the label source explicit, such as manual_review or ground_truth.
For a missed detection, include an unavailable prediction row with an empty `raw_confidence`, `available=false`, and a `missing_reason`, then label that observation as a failure. This preserves misses in coverage accounting without inventing a confidence score.

## Missing raw scores

An absent, non-finite, or unavailable raw score is not calibrated and is never replaced with 1.0 or 0.0. The observation row remains in the confidence JSONL with a null calibrated_confidence, available=false, and a missing_reason. It is excluded from calibration and confidence metrics. A labeled missed detection remains visible as a failed/unscored observation so recall and confidence coverage are not conflated.

If a critical configured entity has no calibrated score, the frame-level q_perception is unavailable. Missing non-critical entities are omitted from the weighted geometric aggregation and their weights are renormalized. Per-camera records remain in the observation JSONL; camera_aggregation controls the task-level aggregation.

## Outputs and comparison

confidence.observations_path stores long-form entity/camera observations, including raw score, calibrated score, labels, split, model/calibrator IDs, and condition metadata. q_perception is stored alongside v and c in the ordinary score sidecar. confidence__<dataset>.json reports held-out test Brier score, ECE, reliability bins, and results by entity and condition.

Supported weighting methods are perception_only, visibility_perception, perception_reliability, and combined_three_way, in addition to the existing baseline methods. Coefficients are normalized over available inputs only when weighting.missing_handling is renormalize; missing confidence remains unavailable in the sidecar.
