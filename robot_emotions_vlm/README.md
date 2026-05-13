# robot_emotions_vlm

VLM-assisted Kimodo motion generation pipeline for the RobotEmotions dataset. Uses `Qwen3-VL-8B-Instruct` to produce text descriptions of video windows, which are then combined with pose-derived spatial anchors to condition the Kimodo diffusion model.

All commands run inside the `kimodo` conda environment.

## Pipeline overview

```
Real video + pose3d → 5-sec windows → Qwen3-VL descriptions
  → build-anchor-catalog (root2d + optional end-effectors from pose3d.npz)
  → generate-kimodo (SMPL-X motion conditioned on text + constraints)
  → export-kimodo-virtual-imu (metric norm → root estimation → IMUSim)
  → export-mixed-virtual-imu (merge real + synthetic for training)
```

## Environment

```bash
conda activate kimodo
python -c "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; import av; print('ok')"
```

Qwen3-VL weights are downloaded from Hugging Face on the first run unless `--local-files-only` is passed.

---

## Commands

### `describe-videos`

Describe full clips with Qwen3-VL and export a Kimodo prompt catalog.

```bash
# Full dataset
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_qwen

# Specific domains
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_qwen

# Single clip
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --clip-id robot_emotions_10ms_u02_tag11 \
  --output-dir output/robot_emotions_qwen_single

# Offline (no HuggingFace download)
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_qwen \
  --local-files-only
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model-id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `--num-video-frames` | `32` | Frames sampled from video |
| `--max-new-tokens` | `384` | Generation budget |
| `--temperature` | `0.2` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling |
| `--system-prompt-path` | built-in | Override system prompt template |
| `--user-prompt-path` | built-in | Override user prompt template |
| `--catalog-output-path` | auto | Custom path for the Kimodo catalog |
| `--seed` | `123` | Generation seed |
| `--num-samples` | `1` | Kimodo samples per clip |

**Outputs:**

- `video_description_manifest.jsonl`
- `video_description_summary.json`
- `kimodo_prompt_catalog.jsonl`
- `<clip_id>/description.json`, `raw_response.txt`, `prompt_context.json`, `quality_report.json`

---

### `describe-windows`

Segment each pose3d clip into overlapping windows, render a short MP4 per window, and describe each with Qwen3-VL. This is the recommended path for Kimodo generation because 5-second windows are within the model's generation range.

Requires a `pose3d_manifest.jsonl` produced by the real-video pipeline (`pose_module.robot_emotions export-pose3d`).

```bash
python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows \
  --window-sec 5.0 \
  --window-hop-sec 2.5 \
  --num-video-frames 48
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--window-sec` | `5.0` | Window duration |
| `--window-hop-sec` | `2.5` | Hop between windows |
| `--max-windows-per-clip` | none | Cap windows per clip |
| `--num-video-frames` | `48` | Frames sampled per window |
| `--clip-id` | all | Process specific clip(s) only |

All Qwen generation flags (`--model-id`, `--temperature`, etc.) are the same as `describe-videos`.

**Outputs:**

- `window_description_manifest.jsonl`
- `window_description_summary.json`
- `kimodo_window_prompt_catalog.jsonl`
- `<prompt_id>/window.mp4`, `description.json`, `raw_response.txt`, `prompt_context.json`, `quality_report.json`

---

### `build-anchor-catalog`

Combines the Qwen window catalog with the real pose3d to build Kimodo-ready constraints. For each window it extracts the ground trajectory (`root2d`) from `pose3d.npz`, rebases it to `x=z=0`, converts to Kimodo's coordinate system, and optionally adds sparse end-effector keyframes.

```bash
# root2d only (default)
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors \
  --model Kimodo-SMPLX-RP-v1

# + end-effector constraints (hands and feet, requires kimodo env)
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors_effectors \
  --model Kimodo-SMPLX-RP-v1 \
  --effector-keyframes 5
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `Kimodo-SMPLX-RP-v1` | Kimodo model for constraint format |
| `--effector-keyframes` | `0` | Sparse end-effector keyframes (0 = root2d only) |
| `--clip-id` | all | Restrict to specific clip(s) |

**Constraint modes:**

- **`root2d` only** (`--effector-keyframes 0`): anchors the ground trajectory; Kimodo freely generates pose from text.
  - Near-static windows (`net_displacement < 0.05 m`) use `stabilized_linear` motion mode.
  - `global_root_heading` is added only when displacement justifies a reliable heading (`>= 0.10 m`).
- **End-effectors** (`--effector-keyframes N`): adds `left-hand`, `right-hand`, `left-foot`, `right-foot` constraints with N uniformly spaced keyframes. Each contains `global_joints_positions` (K, 22, 3). Requires IMUGPT22 → SMPLX22 retargeting.

See [ANCHOR_CATALOG.md](ANCHOR_CATALOG.md) for the full constraint contract.

**Outputs:**

- `kimodo_anchor_catalog.jsonl`
- `kimodo_anchor_catalog.summary.json`
- `<prompt_id>/constraints.json`, `traceability.json`

---

### `generate-kimodo`

Runs Kimodo batch generation for all entries in an anchor catalog.

```bash
# Full catalog
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo

# Single window
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --prompt-id robot_emotions_10ms_u02_tag11__w000 \
  --output-dir output/robot_emotions_kimodo_single
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `Kimodo-SMPLX-RP-v1` | Kimodo model variant |
| `--duration-sec` | `5.0` | Fallback duration when catalog has no hint |
| `--diffusion-steps` | `100` | Diffusion denoising steps |
| `--seed` | catalog value | Override random seed |
| `--num-samples` | catalog value | Samples per prompt |
| `--bvh` | off | Also export BVH (SOMA models) |
| `--cfg-type` | model default | CFG type: `nocfg`, `regular`, `separated` |
| `--cfg-weight` | model default | CFG weight(s) |
| `--clip-id` / `--prompt-id` / `--window-id` | all | Filter entries |

**Outputs:**

- `kimodo_generation_manifest.jsonl`
- `kimodo_generation_summary.json`
- `<prompt_id>/motion.npz` (or `motion/` folder when `num_samples > 1`)
- `<prompt_id>/motion_amass.npz` (for `Kimodo-SMPLX-RP-v1`)
- `<prompt_id>/motion.bvh` (when `--bvh`)
- `<prompt_id>/prompt_entry.json`, `generation_config.json`

---

### `export-kimodo-virtual-imu`

Converts a Kimodo generation manifest to synthetic virtual IMU signals, applying the same normalization and simulation stages as the real-video pipeline (metric normalization → root estimation → IMUSim).

```bash
# Without calibration
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_imu

# With geometric alignment + percentile calibration against real IMU
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_imu \
  --real-imu-root output/exp_real_pose \
  --real-imu-signal-mode acc
```

When `--real-imu-root` is provided, the pipeline resolves `imu.npz` for each window from `reference_clip_id` in the manifest and applies, in order:

1. **Geometric alignment** (`run_geometric_alignment`) — corrects the gravity-axis mismatch between SMPL-X and the physical sensor mount. Controlled by `pose_module/configs/imu_alignment_config.yaml`.
2. **Percentile calibration** (`calibrate_virtual_imu_sequence`) — maps the amplitude distribution of the synthetic signal to the real IMU reference.

Expected real IMU layout: `<real-imu-root>/<domain>/user_<NN>/<clip_id>/imu.npz`

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--real-imu-root` | none | Root of real IMU data; enables calibration |
| `--real-imu-signal-mode` | `acc` | Signal used for calibration: `acc`, `gyro`, `both` |
| `--real-imu-percentile-resolution` | `100` | Percentile bins for rank-mapping |
| `--no-real-imu-per-class-calibration` | off | Use full distribution instead of per-class |
| `--real-imu-label-key` | none | Manifest field for per-class grouping (e.g. `emotion`) |
| `--sensor-layout-path` | none | Custom sensor layout JSON |
| `--imu-acc-noise-std-m-s2` | none | Accelerometer noise override |
| `--imu-gyro-noise-std-rad-s` | none | Gyroscope noise override |
| `--imu-random-seed` | `0` | Noise seed |
| `--export-bvh` | off | Also export BVH per clip |
| `--no-skip-existing` | off | Reprocess even if `virtual_imu.npz` exists |
| `--clip-id` | all | Filter to specific clips |

**Outputs:**

- `virtual_imu_manifest.jsonl`
- `virtual_imu_summary.json`
- `<prompt_id>/virtual_imu/virtual_imu.npz`
- `<prompt_id>/virtual_imu/virtual_imu_calibration_report.json` (when calibration runs)

---

### `export-mixed-virtual-imu`

Merges a real `virtual_imu_manifest.jsonl` and a synthetic one into a single manifest for combined training experiments. No recomputation is performed.

```bash
python -m robot_emotions_vlm export-mixed-virtual-imu \
  --real-manifest output/robot_emotions_virtual_imu/virtual_imu_manifest.jsonl \
  --synthetic-manifest output/robot_emotions_kimodo_imu/virtual_imu_manifest.jsonl \
  --output-dir output/robot_emotions_mixed_imu
```

**Outputs:**

- `mixed_virtual_imu_manifest.jsonl`
- `mixed_virtual_imu_summary.json`

---

## Full pipeline walkthrough

### Step 1 — Export real pose3d (`.venv`)

```bash
./.venv/bin/python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_pose3d
```

### Step 2 — Describe windows (kimodo env)

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows \
  --window-sec 5.0 --window-hop-sec 2.5 --num-video-frames 48
```

### Step 3 — Build anchor catalog (kimodo env)

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors \
  --model Kimodo-SMPLX-RP-v1
```

### Step 4 — Generate motions (kimodo env)

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo
```

### Step 5 — Export virtual IMU (kimodo env)

```bash
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo/kimodo_generation_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_imu \
  --real-imu-root output/exp_real_pose \
  --real-imu-signal-mode acc
```

### Step 6 — Merge real + synthetic (kimodo env)

```bash
python -m robot_emotions_vlm export-mixed-virtual-imu \
  --real-manifest output/robot_emotions_virtual_imu/virtual_imu_manifest.jsonl \
  --synthetic-manifest output/robot_emotions_kimodo_imu/virtual_imu_manifest.jsonl \
  --output-dir output/robot_emotions_mixed_imu
```

---

## Module structure

| File | Role |
|---|---|
| `cli.py` | CLI entry point; defines all subcommands |
| `qwen_backend.py` | Qwen3-VL model loading and inference |
| `windowing.py` | Window segmentation over pose3d trajectories |
| `window_descriptions.py` | `describe-windows` orchestration |
| `anchor_catalog.py` | `build-anchor-catalog` — constraint extraction and retargeting |
| `retarget.py` | IMUGPT22 → SMPLX22 skeleton retargeting |
| `kimodo_generation.py` | `generate-kimodo` batch runner |
| `kimodo_adapter.py` | Kimodo Python API adapter |
| `kimodo_constraints_patch.py` | Constraint format patching for Kimodo versions |
| `schemas.py` | `VideoDescription` dataclass and JSON parsing |
| `prompts.py` | Prompt template loading and rendering |
| `dataset.py` | `RobotEmotionsDataset` — clip enumeration |
| `export.py` | Manifest and artifact writers |
| `metadata.py` | Video metadata extraction |
| `prompt_templates/` | Editable Jinja2 / text prompt templates |

## Notes

- Prompt templates are in `robot_emotions_vlm/prompt_templates/` and can be edited without code changes.
- Manifests are JSONL; each line is a self-contained JSON entry so partial runs can be resumed by filtering on `status`.
- `generate-kimodo` defaults to `Kimodo-SMPLX-RP-v1`, which exports an AMASS-compatible `motion_amass.npz` alongside each motion.
- See [ANCHOR_CATALOG.md](ANCHOR_CATALOG.md) for the full constraint schema and retargeting details.
