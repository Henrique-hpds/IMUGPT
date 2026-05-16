# Pose Module

This module documents how to run the pose pipeline implemented so far. The integrated path currently covers the `RobotEmotions` dataset, spanning the following steps:

- `1 video_loader`
- `2 person_selector`
- `3 vitpose_2d_estimator`
- `4 pose2d_cleaner`
- `5 motionbert_3d_lifter`
- `6 skeleton_mapper`
- `7 metric_normalizer`
- `8 root_trajectory_estimator`
- `9 ik_adapter`
- `10 imusim_adapter`

## What is implemented

Current flow:

- reading the `RobotEmotions` dataset
- discovery of clips and `CSV IMU + video` pairs
- export of `imu.npz` and `metadata.json`
- 2D inference with `ViTPose-B`
- temporal association and main person selection
- 2D temporal cleaning, outlier removal, and adaptation to `motionbert17`
- 3D lifting via sliding windows under the `MotionBERT` contract
- strict mapping `MB17 -> IMUGPT22`
- local metric normalization in the body reference frame
- estimation of `root_translation_m` and pseudo-global composition for downstream
- adaptation of the pseudo-global pose to the IK contract, with BVH and local rotation sequence
- virtual IMU synthesis from the IK result and a `sensor_layout`
- export of `pose/pose2d.npz`, `pose/pose3d.npz`, `pose/pose3d_metric_local.npz`, `pose/ik_sequence.npz`, `pose/virtual_imu.npz`, and auxiliary artifacts

## Module structure

- `pose_module/pipeline.py`: orchestration of steps 1 to 10
- `pose_module/interfaces.py`: contracts and canonical structures
- `pose_module/model_registry.py`: resolution of local models in `pose_module/checkpoints/`
- `pose_module/openmmlab_runtime.py`: shared Python selection from the `openmmlab` env
- `pose_module/download_models.py`: reproducible downloader for weights via `mim`
- `pose_module/io/video_loader.py`: video metadata, fps, and frame selection
- `pose_module/tracking/person_selector.py`: simple tracking and subject selection
- `pose_module/vitpose/estimator.py`: ViTPose backend in the `openmmlab` environment
- `pose_module/vitpose/adapter.py`: conversion to `PoseSequence2D`
- `pose_module/processing/cleaner2d.py`: temporal cleaning and adaptation to `MotionBERT`
- `pose_module/motionbert/adapter.py`: sliding windows and tensor contract `[B, T, J, C]`
- `pose_module/motionbert/lifter.py`: temporal 3D lifting and artifact export for step 5
- `pose_module/processing/skeleton_mapper.py`: deterministic expansion from `motionbert17` to `IMUGPT22`
- `pose_module/processing/metric_normalizer.py`: body reference frame + anthropometric scale + temporal smoothing for step 7
- `pose_module/processing/root_estimator.py`: root trajectory estimation and pseudo-global composition for step 8
- `pose_module/export/ik_adapter.py`: adaptation of the pseudo-global pose to local rotations, offsets, and BVH for step 9
- `pose_module/export/imusim_adapter.py`: virtual IMU synthesis from the IK contract and `sensor_layout` for step 10
- `pose_module/processing/temporal_filters.py`: temporal interpolation and smoothing
- `pose_module/processing/quality.py`: report consolidation
- `pose_module/robot_emotions/extractor.py`: dataset-specific scanner and export
- `pose_module/robot_emotions/pose2d.py`: dataset wrapper over the generic pipeline
- `pose_module/robot_emotions/pose3d.py`: full pipeline export up to step 8
- `pose_module/robot_emotions/virtual_imu.py`: full pipeline export up to step 10
- `pose_module/robot_emotions/cli.py`: current CLI
- `pose_module/configs/sensor_layout.yaml`: default virtual sensor layout

## Prerequisites

- be at the repository root
- use the `.venv` for project Python commands
- have the dataset at `data/RobotEmotions`
- have `ffprobe` available on the system
- have the Conda environment `openmmlab` with Python 3.8 and OpenMMLab (`mmpose`, `mmdet`, `mmpretrain`)
- have the local weights at `pose_module/checkpoints/`

## Step by step

### 1. Validate the Python environments

Use `.venv` for project scripts and the `openmmlab` env for OpenMMLab backends:

```bash
.venv/bin/python -V
conda run -n openmmlab python -V
conda run -n openmmlab python -c "import mmpose, mmdet, mmpretrain; print('openmmlab ok')"
ffprobe -version
```

### 2. Download the local weights from the repository

This step populates `pose_module/checkpoints/` with the models used by the pipeline.

```bash
.venv/bin/python -m pose_module.download_models --env-name openmmlab
find pose_module/checkpoints -maxdepth 1 -type f | sort
```

Expected models:

- `td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192`
- `rtmdet_m_8xb32-300e_coco`
- `motionbert_dstformer-ft-243frm_8xb32-120e_h36m`

### 3. Check the available CLI

```bash
.venv/bin/python -m pose_module.robot_emotions --help
```

The current CLI exposes:

- `scan`
- `export-imu`
- `export-pose2d`
- `export-pose3d`
- `export-virtual-imu`

### 4. List dataset clips

```bash
.venv/bin/python -m pose_module.robot_emotions \
  scan \
  --dataset-root data/RobotEmotions
```

To restrict to specific domains:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  scan \
  --dataset-root data/RobotEmotions \
  --domains 10ms
```

###  Export IMU and metadata only

This step prepares `imu.npz`, `metadata.json`, `manifest.jsonl`, and `summary.json`.

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_extract
```

For a single clip:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_extract \
  --clip-id robot_emotions_10ms_u02_tag05
```

### 6. Run the 2D pipeline via CLI

This command runs the flow up to 2D cleaning, covering steps `1` to `4`.
It ensures `imu.npz` and `metadata.json` are present when needed and saves pose artifacts to `pose/`.

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose2d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose2d \
  --clip-id robot_emotions_10ms_u02_tag05 \
  --env-name openmmlab
```

Without debug video:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose2d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose2d \
  --clip-id robot_emotions_10ms_u02_tag05 \
  --env-name openmmlab \
  --no-debug
```

Remove `--clip-id` to run all clips found in the selected domains.

Backend environment selection (`--env-name`):

- `openmmlab` (default): explicitly uses the Conda env with Python 3.8 and OpenMMLab
- `auto`: tries `openmmlab` first, then falls back to the current Python
- `current`: uses only the current Python
- `<conda_env_name>`: uses only the Python from the specified Conda env

### 7. Run the full 3D pipeline (steps 1 to 8) via CLI

The command below runs the full flow with MotionBERT:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose3d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose3d \
  --clip-id robot_emotions_10ms_u02_tag05 \
  --env-name openmmlab
```

Useful flags for `export-pose3d`:

- `--debug-2d` / `--no-debug-2d`: enables or disables 2D overlays (`debug_overlay.mp4`, `debug_overlay_raw.mp4`, `debug_overlay_clean.mp4`)
- `--debug-3d` / `--no-debug-3d`: enables or disables side-by-side 3D overlays (`debug_overlay_pose3d_raw.mp4`, `debug_overlay_pose3d_imugpt22.mp4`)
- `--no-debug`: disables all debug videos at once
- `--motionbert-env-name <env>`: uses a specific Conda env for the 3D backend
- `--motionbert-window-size <int>`: requested temporal window size
- `--motionbert-window-overlap <float>`: overlap between windows
- `--motionbert-device <device>`: `auto`, `cpu`, `cuda:0`, etc.
- `--no-motionbert-confidence`: removes the confidence channel from the MotionBERT input
- `--allow-motionbert-fallback-backend`: allows falling back to the heuristic backend if the real MotionBERT fails

To generate only the 3D debug output, without 2D videos:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose3d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose3d \
  --clip-id robot_emotions_10ms_u02_tag01 \
  --env-name openmmlab \
  --no-debug-2d \
  --debug-3d
```

This flow generates the 3D artifacts for steps `5`, `6`, `7`, and `8`, with `pose3d.npz` in the final `IMUGPT22` contract under `pseudo_global_metric`, `pose3d_metric_local.npz` preserving the metric local output from step 7, `pose3d_motionbert17.npz` preserving the raw MB17 output from MotionBERT, `3d_keypoints_raw.npy`, `3d_keypoints_metric.npy`, `root_translation.npy`, `motionbert_run.json`, and, when enabled, a side-by-side debug with 2D clean + 3D raw/final.

### 8. Run the full pipeline (steps 1 to 10) via CLI

The command below closes the pipeline with `IK` and `IMUSim`, while preserving the intermediate 3D artifacts:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-virtual-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_virtual_imu_v2_all_dataset \
  --env-name openmmlab \
  --no-debug-2d --no-debug-3d
```

Useful flags for `export-virtual-imu`:

- `--sensor-layout-path <file>`: uses a custom virtual sensor layout
- `--imu-acc-noise-std-m-s2 <float>`: injects Gaussian noise into the accelerometer
- `--imu-gyro-noise-std-rad-s <float>`: injects Gaussian noise into the gyroscope
- `--imu-random-seed <int>`: fixes the seed for optional noise
- `--real-imu-reference-path <file.npz>`: applies virtual-to-real calibration by percentiles using a real reference
- `--real-imu-label-key <field>`: uses a `labels` field from the manifest (`action`, `emotion`, `stimulus`) for per-class calibration
- `--real-imu-signal-mode <acc|gyro|both>`: selects which channels are calibrated against the real reference
- `--real-imu-percentile-resolution <int>`: controls the number of percentiles used in the article mapping
- `--no-real-imu-per-class-calibration`: disables per-class calibration and uses the global distribution of the reference
- `--debug-2d` / `--no-debug-2d`: controls the 2D overlays from the 3D pipeline segment
- `--debug-3d` / `--no-debug-3d`: controls the 3D overlays from the 3D pipeline segment

This flow generates the final artifacts for steps `9` and `10`, including `ik_sequence.npz`, `pose3d_ik.bvh`, `virtual_imu.npz`, `virtual_imu_report.json`, `sensor_layout_resolved.json`, and a consolidated `quality_report.json` covering 3D pose, IK, and virtual IMU. When a real reference is provided, the pipeline also saves `virtual_imu_raw.npz` and `virtual_imu_calibration_report.json`.

The default `sensor_layout` is at `pose_module/configs/sensor_layout.yaml` and replicates the 4-sensor arrangement from `RobotEmotions` (`waist`, `head`, `right_forearm`, `left_forearm`). If your downstream requires a different arrangement, replace this file with the corresponding layout.

### 9. Export BVH via CLI (custom file)

You can export any `pose3d.npz` (including different skeletons such as `motionbert17`) to BVH with a custom output path:

```bash
.venv/bin/python -m pose_module.export.bvh \
  --pose3d-npz output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/pose3d_motionbert17.npz \
  --output-bvh output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/meu_pipeline_3d_raw.bvh
```

CLI arguments:

- `--pose3d-npz`: path to the input file in `PoseSequence3D` format (`.npz`)
- `--output-bvh`: path to the output BVH file
- `--no-ground-to-floor`: optional, disables the vertical adjustment that places the lowest point on the floor

Example to export the final pipeline skeleton (`pose3d.npz`):

```bash
.venv/bin/python -m pose_module.export.bvh \
  --pose3d-npz output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/pose3d.npz \
  --output-bvh output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/pose3d_final.bvh
```

## Generated outputs

### `export-imu` outputs

At the export root:

- `manifest.jsonl`
- `summary.json`

For each exported clip:

- `imu.npz`
- `metadata.json`

Per-clip layout:

```text
<output_dir>/<domain>/user_<id>/<clip_id>/
```

### `export-pose2d` outputs

At the export root:

- `pose_manifest.jsonl`
- `pose_summary.json`

For each exported clip:

- `pose/pose2d.npz`
- `pose/2d_keypoints_raw.npy`
- `pose/2d_keypoints_clean.npy`
- `pose/person_track.json`
- `pose/quality_report.json`
- `pose/backend_run.json`
- `pose/raw_predictions.json`
- `pose/debug_overlay.mp4` when `save_debug=true`
- `pose/debug_overlay_raw.mp4` when `save_debug=true`
- `pose/debug_overlay_clean.mp4` when `save_debug=true`

Per-clip layout:

```text
<output_dir>/<domain>/user_<id>/<clip_id>/pose/
```

### `export-pose3d` outputs

At the export root:

- `pose3d_manifest.jsonl`
- `pose3d_summary.json`

When running the full pipeline up to step `8`, the following files are added to the `pose/` directory:

- `pose/pose3d.npz`
- `pose/pose3d_metric_local.npz`
- `pose/pose3d_motionbert17.npz`
- `pose/pose3d.bvh`
- `pose/3d_keypoints_raw.npy`
- `pose/3d_keypoints_metric.npy`
- `pose/root_translation.npy`
- `pose/motionbert_run.json`
- `pose/debug_overlay_pose3d_raw.mp4` when `save_debug=true` or `save_debug_3d=true`
- `pose/debug_overlay_pose3d_imugpt22.mp4` when `save_debug=true` or `save_debug_3d=true`

### `export-virtual-imu` outputs

At the export root:

- `virtual_imu_manifest.jsonl`
- `virtual_imu_summary.json`

When running the full pipeline up to step `10`, the following files are added to the `pose/` directory:

- all artifacts from `export-pose3d`
- `pose/ik_sequence.npz`
- `pose/ik_report.json`
- `pose/pose3d_ik.bvh`
- `pose/virtual_imu.npz`
- `pose/virtual_imu_report.json`
- `pose/sensor_layout_resolved.json`
- `pose/quality_report.json`

In the 3D pipeline, 2D and 3D overlays can be controlled separately with `save_debug_2d` and `save_debug_3d`.

If a given `Tag` has more than one capture, the extractor generates one record per capture, e.g. `...tag07` and `...tag07_2`.

## Main artifact formats

In `imu.npz`:

- `imu`: `np.ndarray[T, 4, 6]`
- `imu_flat`: `np.ndarray[T, 24]`
- `timestamps_sec`: `np.ndarray[T]`

In `pose/pose2d.npz`:

- `keypoints_xy`: `np.ndarray[T, 17, 2]`
- `confidence`: `np.ndarray[T, 17]`
- `bbox_xywh`: `np.ndarray[T, 4]`
- `frame_indices`: `np.ndarray[T]`
- `timestamps_sec`: `np.ndarray[T]`

In the auxiliary artifacts from step 4:

- `2d_keypoints_raw.npy`: `np.ndarray[T, 17, 2]` in the canonical ViTPose output
- `2d_keypoints_clean.npy`: `np.ndarray[T, 17, 2]` already cleaned and normalized to the `motionbert17` contract

In `pose/pose3d.npz`:

- `joint_positions_xyz`: `np.ndarray[T, 22, 3]`
- `joint_confidence`: `np.ndarray[T, 22]`
- `skeleton_parents`: `np.ndarray[22]`
- `root_translation_m`: `np.ndarray[T, 3]`
- `coordinate_space`: `pseudo_global_metric`

In `pose/pose3d_metric_local.npz`:

- `joint_positions_xyz`: `np.ndarray[T, 22, 3]`
- `joint_confidence`: `np.ndarray[T, 22]`
- `skeleton_parents`: `np.ndarray[22]`
- `coordinate_space`: `body_metric_local`

In `pose/pose3d_motionbert17.npz`:

- `joint_positions_xyz`: `np.ndarray[T, 17, 3]`
- `joint_confidence`: `np.ndarray[T, 17]`
- `skeleton_parents`: `np.ndarray[17]`

In `pose/ik_sequence.npz`:

- `local_joint_rotations`: `np.ndarray[T, 22, 4]` local quaternions under the IK contract
- `root_translation_m`: `np.ndarray[T, 3]`
- `joint_offsets_m`: `np.ndarray[22, 3]`
- `skeleton_parents`: `np.ndarray[22]`
- `joint_names_3d`: `np.ndarray[22]`

In `pose/virtual_imu.npz`:

- `acc`: `np.ndarray[T, S, 3]`
- `gyro`: `np.ndarray[T, S, 3]`
- `sensor_names`: `np.ndarray[S]`
- `timestamps_sec`: `np.ndarray[T]`

In the auxiliary artifacts for steps 5, 6, 7, 8, 9, and 10:

- `3d_keypoints_raw.npy`: `np.ndarray[T, 17, 3]` in camera reference frame
- `3d_keypoints_metric.npy`: `np.ndarray[T, 22, 3]` after local metric normalization and temporal smoothing
- `root_translation.npy`: `np.ndarray[T, 3]` with the estimated pseudo-global pelvis/root trajectory
- `motionbert_run.json`: backend summary, windows, and 3D lifting quality
- `ik_report.json`: IK adaptation summary, including mean reconstruction error
- `pose3d_ik.bvh`: BVH export derived from the final pseudo-global output
- `virtual_imu_report.json`: virtual IMU synthesis summary and sensor dynamic ranges
- `virtual_imu_raw.npz`: raw virtual IMU before calibration against real IMU, when optional calibration is active
- `virtual_imu_calibration_report.json`: percentile calibration summary against the real reference, when active
- `sensor_layout_resolved.json`: sensor layout effectively resolved against the IMUGPT22 skeleton
- `quality_report.json`: final clip quality consolidation, including 3D pose, IK, and virtual IMU
- `debug_overlay_pose3d_raw.mp4`: side-by-side video with the original video + 2D clean pose and the raw 3D pose from MotionBERT
- `debug_overlay_pose3d_imugpt22.mp4`: side-by-side video with the original video + 2D clean pose and the local metric 3D pose on the IMUGPT22 skeleton after step 7

In the `export-pose3d` command, you can control the overlays separately with `--debug-2d` / `--no-debug-2d` and `--debug-3d` / `--no-debug-3d`.

## Notes

- Recommended initial `fps_target`: `20`
- if the original video has a higher fps than the target, the pipeline performs temporal decimation
- if the original video has a lower fps than the target, the pipeline preserves the native fps at this stage
- the current backend uses `ViTPose-B`
- step 5 now first attempts a real MotionBERT backend via `mmpose` in the `openmmlab` env
- the weights used by the pipeline are explicitly stored in `pose_module/checkpoints/`
- to populate this folder on another machine, run `.venv/bin/python -m pose_module.download_models --env-name openmmlab`
- ViTPose 2D uses `td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192`
- the person detector uses `rtmdet_m_8xb32-300e_coco`
- MotionBERT 3D uses `motionbert_dstformer-ft-243frm_8xb32-120e_h36m`
- the heuristic fallback exists only as an option and is disabled by default
- the default flow exports all clips found in the selected domains
- to restrict execution, use `--clip-id` or `--domains`
- `export-pose3d` ends at step `8`, preserving the final pseudo-global pose
- `export-virtual-imu` closes the pipeline up to `10`, reusing the same final 3D output
