# Framework for generating synthetic IMU based on Human Poses

Framework for generating synthetic Inertial Measurement Unit (IMU) data from human motion. It bridges two domains that are rarely connected: pose estimation and motion generation on one side, and wearable sensor simulation on the other. Given a video of a person moving, or a plain-text description of a movement, the framework produces synthetic accelerometer readings as if physical sensors had been attached to the body.

The primary motivation is the scarcity of labelled IMU datasets. Collecting real IMU data requires physical hardware, synchronized recordings, and careful placement of sensors — a process that is slow, expensive, and hard to scale. IMUGPT addresses this by deriving IMU signals from 3D skeletal trajectories, which can themselves be obtained either from video (via pose estimation) or generated from text prompts (via language-conditioned motion models). This makes it possible to produce large, diverse, and controllable datasets for training and evaluating activity recognition models without any physical sensor.

The framework integrates two main pipelines. The first takes a monocular video, lifts the detected 2D keypoints to a metric 3D skeleton using MotionBERT, and feeds the resulting trajectory into IMUSim to synthesize the sensor signals. The second pipeline uses a Vision-Language Model (Qwen3-VL) to automatically describe short windows of real motion and feeds those descriptions — together with real joint positions as anchors — into Kimodo, a SMPL-X motion generator, to produce novel but physically grounded synthetic motions.

This project was developed in a partnership between the Cognitive Architectures research line from the [**Hub for Artificial Intelligence and Cognitive Architectures (H.IAAC)**](https://h-iaac.github.io/HIAAC-Index) from State University of Campinas (UNICAMP), Brazil; and the [**Robotics and Artificial Inteligence Lab (AIRLab)**](airlab.deib.polimi.it/), from Politecnico di Milano (POLIMI), Italy. 

<!--Badges-->
<!--Meta 1: Arquiteturas Cognitivas-->
[![](https://img.shields.io/badge/-H.IAAC-eb901a?style=for-the-badge&labelColor=black)](https://hiaac.unicamp.br/)[![](https://img.shields.io/badge/-Arq.Cog-black?style=for-the-badge&labelColor=white&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4gPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1Ni4wMDQiIGhlaWdodD0iNTYiIHZpZXdCb3g9IjAgMCA1Ni4wMDQgNTYiPjxwYXRoIGlkPSJhcnFjb2ctMiIgZD0iTTk1NS43NzQsMjc0LjJhNi41Nyw2LjU3LDAsMCwxLTYuNTItNmwtLjA5MS0xLjE0NS04LjEtMi41LS42ODksMS4xMjNhNi41NCw2LjU0LDAsMCwxLTExLjEzNi4wMjEsNi41Niw2LjU2LDAsMCwxLDEuMzY4LTguNDQxbC44LS42NjUtMi4xNS05LjQ5MS0xLjIxNy0uMTJhNi42NTUsNi42NTUsMCwwLDEtMi41OS0uODIyLDYuNTI4LDYuNTI4LDAsMCwxLTIuNDQzLTguOSw2LjU1Niw2LjU1NiwwLDAsMSw1LjctMy4zLDYuNDU2LDYuNDU2LDAsMCwxLDIuNDU4LjQ4M2wxLC40MSw2Ljg2Ny02LjM2Ni0uNDg4LTEuMTA3YTYuNTMsNi41MywwLDAsMSw1Ljk3OC05LjE3Niw2LjU3NSw2LjU3NSwwLDAsMSw2LjUxOCw2LjAxNmwuMDkyLDEuMTQ1LDguMDg3LDIuNS42ODktMS4xMjJhNi41MzUsNi41MzUsMCwxLDEsOS4yODksOC43ODZsLS45NDcuNjUyLDIuMDk1LDkuMjE4LDEuMzQzLjAxM2E2LjUwNyw2LjUwNywwLDAsMSw1LjYwOSw5LjcyMSw2LjU2MSw2LjU2MSwwLDAsMS01LjcsMy4zMWgwYTYuNCw2LjQsMCwwLDEtMi45ODctLjczMmwtMS4wNjEtLjU1LTYuNjgsNi4xOTIuNjM0LDEuMTU5YTYuNTM1LDYuNTM1LDAsMCwxLTUuNzI1LDkuNjkxWm0wLTExLjQ2MWE0Ljk1LDQuOTUsMCwxLDAsNC45NTIsNC45NUE0Ljk1Nyw0Ljk1NywwLDAsMCw5NTUuNzc0LDI2Mi43MzlaTTkzNC44LDI1Ny4zMjVhNC45NTIsNC45NTIsMCwxLDAsNC4yMjEsMi4zNDVBNC45Myw0LjkzLDAsMCwwLDkzNC44LDI1Ny4zMjVabS0uMDIyLTEuNThhNi41MTQsNi41MTQsMCwwLDEsNi41NDksNi4xTDk0MS40LDI2M2w4LjA2MSwyLjUuNjg0LTEuMTQ1YTYuNTkxLDYuNTkxLDAsMCwxLDUuNjI0LTMuMjA2LDYuNDQ4LDYuNDQ4LDAsMCwxLDIuODQ0LjY1bDEuMDQ5LjUxOSw2LjczNC02LjI1MS0uNTkzLTEuMTQ1YTYuNTI1LDYuNTI1LDAsMCwxLC4xMTUtNi4yMjksNi42MTgsNi42MTgsMCwwLDEsMS45NjYtMi4xMzRsLjk0NC0uNjUyLTIuMDkzLTkuMjIyLTEuMzM2LS4wMThhNi41MjEsNi41MjEsMCwwLDEtNi40MjktNi4xbC0uMDc3LTEuMTY1LTguMDc0LTIuNS0uNjg0LDEuMTQ4YTYuNTM0LDYuNTM0LDAsMCwxLTguOTY2LDIuMjY0bC0xLjA5MS0uNjUyLTYuNjE3LDYuMTMxLjc1MSwxLjE5MmE2LjUxOCw2LjUxOCwwLDAsMS0yLjMsOS4xNjRsLTEuMS42MTksMi4wNiw5LjA4NywxLjQ1MS0uMUM5MzQuNDc1LDI1NS43NSw5MzQuNjI2LDI1NS43NDQsOTM0Ljc3OSwyNTUuNzQ0Wm0zNi44NDQtOC43NjJhNC45NzcsNC45NzcsMCwwLDAtNC4zMTYsMi41LDQuODg5LDQuODg5LDAsMCwwLS40NjQsMy43NjIsNC45NDgsNC45NDgsMCwxLDAsNC43NzktNi4yNjZaTTkyOC43LDIzNS41MzNhNC45NzksNC45NzksMCwwLDAtNC4zMTcsMi41LDQuOTQ4LDQuOTQ4LDAsMCwwLDQuMjkxLDcuMzkxLDQuOTc1LDQuOTc1LDAsMCwwLDQuMzE2LTIuNSw0Ljg4Miw0Ljg4MiwwLDAsMCwuNDY0LTMuNzYxLDQuOTQsNC45NCwwLDAsMC00Ljc1NC0zLjYzWm0zNi43NzYtMTAuMzQ2YTQuOTUsNC45NSwwLDEsMCw0LjIyMiwyLjM0NUE0LjkyMyw0LjkyMywwLDAsMCw5NjUuNDc5LDIyNS4xODdabS0yMC45NTItNS40MTVhNC45NTEsNC45NTEsMCwxLDAsNC45NTEsNC45NTFBNC45NTcsNC45NTcsMCwwLDAsOTQ0LjUyNywyMTkuNzcyWiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTkyMi4xNDMgLTIxOC4yKSIgZmlsbD0iIzgzMDNmZiI+PC9wYXRoPjwvc3ZnPiA=)](https://h-iaac.github.io/HIAAC-Index)

<!-- POLIMI / AIRLAB -->
[![](https://img.shields.io/badge/-DEIB.POLIMI-102c53?style=for-the-badge&labelColor=black)](https://www.deib.polimi.it/eng/home-page)[![](https://img.shields.io/badge/-AIRLab-white?style=for-the-badge&labelColor=white&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/Pgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDIwMDEwOTA0Ly9FTiIKICJodHRwOi8vd3d3LnczLm9yZy9UUi8yMDAxL1JFQy1TVkctMjAwMTA5MDQvRFREL3N2ZzEwLmR0ZCI+CjxzdmcgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiB3aWR0aD0iMjUwLjAwMDAwMHB0IiBoZWlnaHQ9IjI1MC4wMDAwMDBwdCIgdmlld0JveD0iMCAwIDI1MC4wMDAwMDAgMjUwLjAwMDAwMCIKIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgoKPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsMjUwLjAwMDAwMCkgc2NhbGUoMC4xMDAwMDAsLTAuMTAwMDAwKSIKZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSJub25lIj4KPHBhdGggZD0iTTAgMTI1MCBsMCAtMTI1MCAxMjUwIDAgMTI1MCAwIDAgMTI1MCAwIDEyNTAgLTEyNTAgMCAtMTI1MCAwIDAKLTEyNTB6IG0xNTM1IDEwNTMgYzMgLTEwIDM1IC05OSA3MCAtMTk4IDM1IC05OSA3NSAtMjExIDg5IC0yNTAgMTMgLTM4IDE0OAotNDIxIDMwMCAtODUwIDE1MSAtNDI5IDI3OCAtNzg4IDI4MSAtNzk4IDcgLTE2IC03IC0xNyAtMjQxIC0xNSBsLTI0OSAzIC00OAoxNTUgYy0zOCAxMjQgLTExOCAzOTQgLTI1NCA4NjMgLTMgOSAtOSAxNyAtMTMgMTcgLTQgMCAtMzIgLTI4IC02MiAtNjIgLTI5Ci0zNCAtMTE2IC0xMzUgLTE5MyAtMjIzIC03NyAtODkgLTE5MyAtMjIyIC0yNTcgLTI5NiAtMTYzIC0xODcgLTE1NiAtMTc2Ci0xOTggLTMxMyAtMjEgLTY5IC00MSAtMTI5IC00NCAtMTM1IC0xMSAtMTcgLTQ4NiAtMTUgLTQ4NiAyIDAgOCA1MCAxNTUgMTExCjMyOCAxMTYgMzMwIDM5MyAxMTE3IDUzNSAxNTI0IDQ2IDEzMiA4NiAyNDYgODkgMjUzIDMgOSA2OSAxMiAyODQgMTIgMjU2IDAKMjc5IC0xIDI4NiAtMTd6Ii8+CjxwYXRoIGQ9Ik0xMjQ2IDE5NTAgYy0zIC04IC0xMiAtNDIgLTIyIC03NSAtMTggLTY5IC0xMjAgLTQwNiAtMjAwIC02NjcgLTMwCi05NyAtNTQgLTE4MSAtNTQgLTE4NyAwIC02IDExMCA0NCAyNDUgMTExIDE0OSA3NSAyNDUgMTI4IDI0NSAxMzcgMCA4IC0zNgoxMzAgLTgwIDI3MCAtNDMgMTQxIC05MCAyOTQgLTEwNCAzNDEgLTEzIDQ3IC0yNyA3OCAtMzAgNzB6Ii8+CjwvZz4KPC9zdmc+Cg==)](https://airlab.deib.polimi.it/)

## Repository Structure

- `pose_module/` — core pipeline: 2D/3D pose estimation, virtual IMU synthesis, MotionBERT lifting. Runs in the `pose_module` conda env.
- `robot_emotions_vlm/` — Qwen3-VL video description, anchor catalog construction, and Kimodo batch generation. Runs in the `kimodo` conda env.
- `kimodo/` — git submodule: SMPL-X motion generator CLI (`kimodo_gen`, `kimodo_textencoder`).
- `imusim/` — IMU physics simulation library used to synthesize accelerometer/gyroscope readings from 3D skeleton trajectories.
- `data/` — input datasets (e.g. `data/RobotEmotions/`).
- `output/` — per-clip outputs organized by experiment; manifests (JSONL) index all artifacts.
- `evaluation/` — notebooks and scripts for classifier experiments and IMU quality assessment.
- `scripts/` — utility scripts.

## Dependencies / Requirements

**Requirements:** Linux, CUDA-capable GPU (≥ 21 GB VRAM recommended).

This project uses `Miniconda` to manage Python environments. We do not recommend `python-env` or `venv`, because the project requires multiple envs with different Python versions.

Make sure to install it accordingly to your Linux distribution. If not, follow the [official instructions here](https://www.anaconda.com/docs/getting-started/miniconda/install/overview).

Also, install `ffmpeg` for your distro. For Ubuntu/Debian, use:

```bash
sudo apt update && sudo apt install ffmpeg -y
```

Now, clone the project's repository. For using the necessary 3th-party codes, use the `--recurse-submodules` flag:  

```bash
git clone --recurse-submodules git@github.com:H-IAAC/POSE2IMU-Framework.git
cd POSE2IMU-Framework
```

## Installation / Usage

All environments (`pose_module`, `openmmlab`, `kimodo`) are configured by a single script:

```bash
sudo chmod +x config_envs.sh
bash config_envs.sh
```

The script creates and installs each conda environment in order, printing progress for each step. If any step fails it stops immediately.

Qwen3-VL weights (`Qwen/Qwen3-VL-8B-Instruct`) are downloaded automatically from Hugging Face on first use.

The project supports two main pipelines. All commands are run from the repository root.

---

### Pipeline 1 — Video → Virtual IMU

Converts real video recordings into synthetic IMU data. The `pose_module` drives the full chain: 2D detection (OpenMMlab/ViTPose), 3D lifting (MotionBERT), metric normalization, root estimation, and physics-based IMU synthesis (IMUSim).

```bash
# Export 3D poses from video
conda run -n pose_module python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_pose3d \
  --fps-target 20

# Synthesize virtual IMU signals from 3D poses
conda run -n pose_module python -m pose_module.robot_emotions export-virtual-imu \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_virtual_imu \
  --fps-target 20
```

Outputs per clip under `output/<experiment>/<clip_id>/`:
- `pose/pose3d/pose3d.npz` — 3D skeleton trajectory
- `imu/virtual_imu.npz` — synthetic accelerometer + gyroscope

---

### Pipeline 2 — Window-Anchored Kimodo Generation

Uses real video windows as anchors: Qwen3-VL describes each 5-second segment; Kimodo generates new SMPL-X motion conditioned on the text and real joint positions.

**Step 1 — Export real pose3d** (same as Pipeline 1, `pose_module` env):

```bash
conda run -n pose_module python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions --domains 10ms 30ms \
  --output-dir output/robot_emotions_pose3d --fps-target 20
```

**Step 2 — Describe windows with Qwen3-VL** (`kimodo` env):

```bash
conda run -n kimodo python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows
```

**Step 3 — Build anchor catalog** (`kimodo` env):

```bash
conda run -n kimodo python -m robot_emotions_vlm build-anchor-catalog \
  --window-manifest-path output/robot_emotions_qwen_windows/window_description_manifest.jsonl \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors
```

**Step 4 — Generate with Kimodo** (`kimodo` env):

```bash
conda run -n kimodo python -m robot_emotions_vlm generate-kimodo \
  --model Kimodo-SMPLX-RP-v1 \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo
```

Each generated clip produces `motion.npz` + `motion_amass.npz` (SMPL-X) under `output/robot_emotions_kimodo/<clip_id>/`.

---

### Direct Kimodo generation (optional, `kimodo` env)

```bash
conda run -n kimodo kimodo_gen "A person sits down and stands up" \
  --model Kimodo-SMPLX-RP-v1 --duration 10.0 --output output/kimodo_direct/
```

## Citation

<!--Don't remove the following tags, it's used for placing the generated citation from the CFF file-->
<!--CITATION START-->
```bibtex
@software{POSE2IMU,
author = {Parede, Henrique and Bonarini, Andrea and Dornhofer Paro Costa, Paula},
title = { POSE2IMU-Framework},
url = {https://github.com/H-IAAC/POSE2IMU-Framework}
}
```
<!--CITATION END-->

## Authors
  
- (2026-) [Henrique Parede](https://github.com/Henrique-hpds): Computer Engineering student, FEEC-UNICAMP
- (Advisor, 2026-) Andrea Bonarini: Professor, DEIB-POLIMI
- (Advisor, 2026-) Paula Dornhofer Paro Costa: Professor, FEEC-UNICAMP
  
## Acknowledgements

This study was financed by the São Paulo Research Foundation (FAPESP), Brasil. Process Number 2025/21964-5.

This codebase was developed starting from a fork of [**IMUGPT**](https://github.com/ZikangLeng/IMUGPT) by Leng et al. We are grateful for their foundational work, which made this project possible.

We also gratefully acknowledge the authors of the following open-source projects, which are integrated as submodules:

- [**Kimodo**](https://github.com/nv-tlabs/kimodo) (NVIDIA) - SMPL-X motion generation conditioned on text and pose anchors.
- [**ST-GCN**](https://github.com/yysijie/st-gcn) (Sijie Yan et al.) - Spatial Temporal Graph Convolutional Networks for skeleton-based action recognition.
- [**TS2Vec**](https://github.com/zhihanyue/ts2vec) (Zhihan Yue et al.) - Unsupervised time-series representation learning.
