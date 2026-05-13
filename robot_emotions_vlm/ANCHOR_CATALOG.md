# Anchor Catalog

Documentation for the `robot_emotions_vlm.anchor_catalog` module and the window-level Qwen → Kimodo flow.

## What it does

The `build-anchor-catalog` command creates a per-window catalog from:

- `pose3d_manifest.jsonl` exported from the real branch
- `kimodo_window_prompt_catalog.jsonl` exported by Qwen per window

For each window, it:

- selects the exact time window in `pose3d.npz`
- interpolates the 2D root trajectory (`root_translation_m`) onto the Kimodo frame grid
- rebases so the first root starts at `x=z=0` and converts to Kimodo's coordinate system (`-X`, `-Z`)
- writes `constraints.json` with dense `root2d` and, optionally, sparse end-effectors
- saves `traceability.json`
- writes an entry in `kimodo_anchor_catalog.jsonl`

## Current contract

By default (`--effector-keyframes 0`), the only constraint is `root2d`-only: it anchors the ground trajectory from the real capture and lets the model freely generate the pose from the text prompt.

- dense `root2d` (one point per Kimodo frame), interpolated from the real trajectory
- near-static windows (`root2d_net_displacement_m < 0.05`) use `root2d_motion_mode = stabilized_linear`
- `global_root_heading` is added only when the net displacement justifies a reliable heading (`>= 0.10 m`)
- no pose retargeting is performed; no dependency on the `kimodo` conda env in `build-anchor-catalog`

With `--effector-keyframes N` (N > 0), four sparse end-effector constraints are added:

- `left-hand`, `right-hand`, `left-foot`, `right-foot` with N uniformly spaced keyframes
- these are the constraint types for which Kimodo's diffusion model was trained,
  ensuring temporally coherent interpolation between anchor keyframes
- requires IMUGPT22 → SMPLX22 retargeting: corrects hip bone direction inversion (L_Hip, R_Hip)
  and rescales bone lengths to canonical SMPLX22 values
- each constraint includes `global_joints_positions` (K, 22, 3) — Kimodo selects the relevant joint
  internally via `joint_names`
- requires the `kimodo` conda environment

In one sentence: the catalog anchors the ground trajectory from the real capture while Kimodo generates
plausible pose variation from the text prompt, with the option to also anchor the
extremities (hands and feet) to preserve the character of the original movement.

## Recommended defaults

To maintain good temporal coverage:

- `describe-windows --window-sec 5.0`
- `describe-windows --window-hop-sec 2.5`
- `describe-windows --num-video-frames 48`

Fixed configuration for this version:

- `root2d_min_displacement_m = 0.05`
- `heading_min_displacement_m = 0.10`

`build-anchor-catalog` uses `root2d`-only by default. Use `--effector-keyframes N` to add
end-effectors at the extremities (requires `kimodo` env). A value of 5–10 keyframes is sufficient
for 5-second windows at 20 fps.

## How to run

### 1. Export real pose3d

In the project `.venv`:

```bash
./.venv/bin/python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose3d \
  --env-name openmmlab \
  --motionbert-device cuda:0 \
  --no-debug
```

### 2. Generate the Qwen textual catalog per window

In the `kimodo` environment:

```bash
python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows \
  --window-sec 5.0 \
  --window-hop-sec 2.5 \
  --num-video-frames 48
```

### 3. Build the anchored catalog

Default mode (`root2d` only):

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors \
  --model Kimodo-SMPLX-RP-v1
```

With end-effectors at the extremities (requires `kimodo` env):

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors_effectors \
  --model Kimodo-SMPLX-RP-v1 \
  --effector-keyframes 5
```

The resulting `constraints.json` saves:

- `root2d.frame_indices`, `root2d.smooth_root_2d`, `root2d.global_root_heading` when applicable
- `left-hand`, `right-hand`, `left-foot`, `right-foot` with `global_joints_positions` (K, 22, 3)
  when `--effector-keyframes > 0`

### 4. Generate motions with anchors

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors_effectors/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_generated
```

To iterate on a specific window:

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors_effectors/kimodo_anchor_catalog.jsonl \
  --prompt-id robot_emotions_30ms_u03_tag07__w000 \
  --output-dir output/robot_emotions_kimodo_generated_single
```

### 5. Convert Kimodo SMPL-X to pipeline pose3d

After `generate-kimodo`, the motions are in SMPL-X format (motion.npz). To compare them with the real pose3d,
convert them to the pipeline's normalized coordinate system:

```bash
python scripts/batch_kimodo_pose3d.py \
  --manifest output/robot_emotions_kimodo_generated/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_pose3d
```

This:
- Reads each `motion.npz` generated by Kimodo
- Applies `metric_normalizer` (normalizes bone scale) and `root_estimator` (estimates global root)
- Produces `pose3d.npz` in each `<prompt_id>/` subdirectory, with the same structure as the real pose3d
- Generates `kimodo_pose3d_manifest.jsonl` referencing all outputs

Optional: use `--skip-existing` (default) to skip already-processed windows.

### 6. Export virtual IMU

Without alignment (previous behavior):

```bash
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo_generated/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_imu
```

With geometric alignment + percentile calibration (recommended):

```bash
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo_generated_single/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_imu \
  --real-imu-root output/exp_real_pose \
  --real-imu-signal-mode acc
```

The `--real-imu-root` flag automatically resolves the `imu.npz` for each window from the
`reference_clip_id` in the Kimodo manifest. For each window it applies, in order:

1. Geometric alignment (`run_geometric_alignment`) — corrects the frame mismatch between the
   simulated sensor (gravity on SMPL-X Y axis) and the physical sensor (fixed mount on the arm).
   Controlled by `pose_module/configs/imu_alignment_config.yaml`.
2. Percentile calibration (`calibrate_virtual_imu_sequence`) — maps the amplitude distribution of the
   synthetic IMU to that of the real IMU from the reference clip.

Additional flags:

```
--real-imu-signal-mode acc|gyro|both    Signal used for calibration (default: acc)
--real-imu-percentile-resolution N      Percentile bins (default: 100)
--no-real-imu-per-class-calibration     Disables per-emotion-class calibration
--real-imu-label-key FIELD              Manifest field for per-class calibration (e.g.: emotion)
```

## Main outputs

In `build-anchor-catalog`:

- `kimodo_anchor_catalog.jsonl`
- `kimodo_anchor_catalog.summary.json`
- `<prompt_id>/constraints.json`
- `<prompt_id>/traceability.json`

In `describe-windows`:

- `window_description_manifest.jsonl`
- `window_description_summary.json`
- `kimodo_window_prompt_catalog.jsonl`
- `<prompt_id>/window.mp4`

In `generate-kimodo`:

- `kimodo_generation_manifest.jsonl`
- `kimodo_generation_summary.json`
- `<prompt_id>/motion.npz` or `motion/` folder when `num_samples > 1`
- `motion_amass.npz` for `Kimodo-SMPLX-RP-v1`

In `batch_kimodo_pose3d.py`:

- `kimodo_pose3d_manifest.jsonl`
- `kimodo_pose3d_summary.json`
- `<prompt_id>/pose3d.npz` — normalized pose3d, comparable with the real pose3d
- `<prompt_id>/error_trace.txt` — if a processing error occurs
