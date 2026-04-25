# Framework for generating synthetic IMU based on Human Poses

Framework for generating synthetic Inertial Measurement Unit (IMU) data from human motion. It bridges two domains that are rarely connected: pose estimation and motion generation on one side, and wearable sensor simulation on the other. Given a video of a person moving, or a plain-text description of a movement, the framework produces synthetic accelerometer readings as if physical sensors had been attached to the body.

The primary motivation is the scarcity of labelled IMU datasets. Collecting real IMU data requires physical hardware, synchronized recordings, and careful placement of sensors — a process that is slow, expensive, and hard to scale. IMUGPT addresses this by deriving IMU signals from 3D skeletal trajectories, which can themselves be obtained either from video (via pose estimation) or generated from text prompts (via language-conditioned motion models). This makes it possible to produce large, diverse, and controllable datasets for training and evaluating activity recognition models without any physical sensor.

The framework integrates three independent pipelines. The first takes a monocular video, lifts the detected 2D keypoints to a metric 3D skeleton using MotionBERT, and feeds the resulting trajectory into IMUSim to synthesize the sensor signals. The second pipeline replaces the video with a natural-language prompt: T2M-GPT generates a 3D motion sequence from text, which then goes through the same IMU synthesis chain. The third pipeline uses a Vision-Language Model (Qwen3-VL) to automatically describe short windows of real motion and feeds those descriptions — together with real joint positions as anchors — into Kimodo, a SMPL-X motion generator, to produce novel but physically grounded synthetic motions.

This project was developed as part of the Cognitive Architectures research line from the Hub for Artificial Intelligence and Cognitive Architectures (H.IAAC) of the State University of Campinas (UNICAMP).
See more projects from the group [here](https://h-iaac.github.io/HIAAC-Index).

<!--Badges-->
[![](https://img.shields.io/badge/-H.IAAC-eb901a?style=for-the-badge&labelColor=black)](https://hiaac.unicamp.br/)

<!--Meta 1: Arquiteturas Cognitivas-->
[![](https://img.shields.io/badge/-Arq.Cog-black?style=for-the-badge&labelColor=white&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4gPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1Ni4wMDQiIGhlaWdodD0iNTYiIHZpZXdCb3g9IjAgMCA1Ni4wMDQgNTYiPjxwYXRoIGlkPSJhcnFjb2ctMiIgZD0iTTk1NS43NzQsMjc0LjJhNi41Nyw2LjU3LDAsMCwxLTYuNTItNmwtLjA5MS0xLjE0NS04LjEtMi41LS42ODksMS4xMjNhNi41NCw2LjU0LDAsMCwxLTExLjEzNi4wMjEsNi41Niw2LjU2LDAsMCwxLDEuMzY4LTguNDQxbC44LS42NjUtMi4xNS05LjQ5MS0xLjIxNy0uMTJhNi42NTUsNi42NTUsMCwwLDEtMi41OS0uODIyLDYuNTI4LDYuNTI4LDAsMCwxLTIuNDQzLTguOSw2LjU1Niw2LjU1NiwwLDAsMSw1LjctMy4zLDYuNDU2LDYuNDU2LDAsMCwxLDIuNDU4LjQ4M2wxLC40MSw2Ljg2Ny02LjM2Ni0uNDg4LTEuMTA3YTYuNTMsNi41MywwLDAsMSw1Ljk3OC05LjE3Niw2LjU3NSw2LjU3NSwwLDAsMSw2LjUxOCw2LjAxNmwuMDkyLDEuMTQ1LDguMDg3LDIuNS42ODktMS4xMjJhNi41MzUsNi41MzUsMCwxLDEsOS4yODksOC43ODZsLS45NDcuNjUyLDIuMDk1LDkuMjE4LDEuMzQzLjAxM2E2LjUwNyw2LjUwNywwLDAsMSw1LjYwOSw5LjcyMSw2LjU2MSw2LjU2MSwwLDAsMS01LjcsMy4zMWgwYTYuNCw2LjQsMCwwLDEtMi45ODctLjczMmwtMS4wNjEtLjU1LTYuNjgsNi4xOTIuNjM0LDEuMTU5YTYuNTM1LDYuNTM1LDAsMCwxLTUuNzI1LDkuNjkxWm0wLTExLjQ2MWE0Ljk1LDQuOTUsMCwxLDAsNC45NTIsNC45NUE0Ljk1Nyw0Ljk1NywwLDAsMCw5NTUuNzc0LDI2Mi43MzlaTTkzNC44LDI1Ny4zMjVhNC45NTIsNC45NTIsMCwxLDAsNC4yMjEsMi4zNDVBNC45Myw0LjkzLDAsMCwwLDkzNC44LDI1Ny4zMjVabS0uMDIyLTEuNThhNi41MTQsNi41MTQsMCwwLDEsNi41NDksNi4xTDk0MS40LDI2M2w4LjA2MSwyLjUuNjg0LTEuMTQ1YTYuNTkxLDYuNTkxLDAsMCwxLDUuNjI0LTMuMjA2LDYuNDQ4LDYuNDQ4LDAsMCwxLDIuODQ0LjY1bDEuMDQ5LjUxOSw2LjczNC02LjI1MS0uNTkzLTEuMTQ1YTYuNTI1LDYuNTI1LDAsMCwxLC4xMTUtNi4yMjksNi42MTgsNi42MTgsMCwwLDEsMS45NjYtMi4xMzRsLjk0NC0uNjUyLTIuMDkzLTkuMjIyLTEuMzM2LS4wMThhNi41MjEsNi41MjEsMCwwLDEtNi40MjktNi4xbC0uMDc3LTEuMTY1LTguMDc0LTIuNS0uNjg0LDEuMTQ4YTYuNTM0LDYuNTM0LDAsMCwxLTguOTY2LDIuMjY0bC0xLjA5MS0uNjUyLTYuNjE3LDYuMTMxLjc1MSwxLjE5MmE2LjUxOCw2LjUxOCwwLDAsMS0yLjMsOS4xNjRsLTEuMS42MTksMi4wNiw5LjA4NywxLjQ1MS0uMUM5MzQuNDc1LDI1NS43NSw5MzQuNjI2LDI1NS43NDQsOTM0Ljc3OSwyNTUuNzQ0Wm0zNi44NDQtOC43NjJhNC45NzcsNC45NzcsMCwwLDAtNC4zMTYsMi41LDQuODg5LDQuODg5LDAsMCwwLS40NjQsMy43NjIsNC45NDgsNC45NDgsMCwxLDAsNC43NzktNi4yNjZaTTkyOC43LDIzNS41MzNhNC45NzksNC45NzksMCwwLDAtNC4zMTcsMi41LDQuOTQ4LDQuOTQ4LDAsMCwwLDQuMjkxLDcuMzkxLDQuOTc1LDQuOTc1LDAsMCwwLDQuMzE2LTIuNSw0Ljg4Miw0Ljg4MiwwLDAsMCwuNDY0LTMuNzYxLDQuOTQsNC45NCwwLDAsMC00Ljc1NC0zLjYzWm0zNi43NzYtMTAuMzQ2YTQuOTUsNC45NSwwLDEsMCw0LjIyMiwyLjM0NUE0LjkyMyw0LjkyMywwLDAsMCw5NjUuNDc5LDIyNS4xODdabS0yMC45NTItNS40MTVhNC45NTEsNC45NTEsMCwxLDAsNC45NTEsNC45NTFBNC45NTcsNC45NTcsMCwwLDAsOTQ0LjUyNywyMTkuNzcyWiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTkyMi4xNDMgLTIxOC4yKSIgZmlsbD0iIzgzMDNmZiI+PC9wYXRoPjwvc3ZnPiA=)](https://h-iaac.github.io/HIAAC-Index)

> As seguintes seções são obrigatórias, devendo começar com um cabeçalho de nível 2 (## Nome da seção), mantendo o nome delas. No caso de nomes condicionais (A / B), é possível manter assim ou escolher apenas um nome (## A / B ou ## A ou ## B)
>
> Caso deseje, pode adicionar outras seções além das obrigatórias (utilize apenas cabeçalhos de nível 2 e maior)

## Repository Structure
> Lista e descrição das pastas e arquivos importantes na raiz do repositório

- `pose_module/` — core pipeline: 2D/3D pose estimation, virtual IMU synthesis, MotionBERT lifting, T2M-GPT text-to-pose backend. Runs in `.venv`.
- `robot_emotions_vlm/` — Qwen3-VL video description, anchor catalog construction, and Kimodo batch generation. Runs in the `kimodo` conda env.
- `kimodo/` — git submodule: SMPL-X motion generator CLI (`kimodo_gen`, `kimodo_textencoder`).
- `imusim/` — IMU physics simulation library used to synthesize accelerometer/gyroscope readings from 3D skeleton trajectories.
- `t2mgpt/` — T2M-GPT VQ-VAE weights and inference code for text-driven pose generation.
- `data/` — input datasets (e.g. `data/RobotEmotions/`).
- `output/` — per-clip outputs organized by experiment; manifests (JSONL) index all artifacts.
- `evaluation/` — notebooks and scripts for classifier experiments and IMU quality assessment.
- `scripts/` — utility scripts.


## Dependencies / Requirements

**Requirements:** Linux, CUDA-capable GPU (≥ 20 GB VRAM recommended).

This project uses `Miniconda` to create the Python enviroments. We do not recommend to use `python-env`, because it will be necessary to configure multiple envs with diferent Python3 versions.

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

All three environments (`pose_module`, `openmmlab`, `kimodo`) are configured by a single script:

```bash
bash config_envs.sh
```

The script creates and installs each conda environment in order, printing progress for each step. If any step fails it stops immediately.

Pre-trained weights for T2M-GPT must be placed manually after running the script:
- `pretrained/VQVAEV3_CB1024_CMT_H1024_NRES3/` — VQ-VAE decoder
- `checkpoints/t2m/` — mean/std normalization stats

Qwen3-VL weights (`Qwen/Qwen3-VL-8B-Instruct`) are downloaded automatically from Hugging Face on first use.

The project supports three independent pipelines. All commands are run from the repository root.

---

### Pipeline 1 — Video → Virtual IMU

Converts real video recordings into synthetic IMU data. The `pose_module` drives the full chain: 2D detection (OpenMMlab/ViTPose), 3D lifting (MotionBERT), metric normalization, root estimation, and physics-based IMU synthesis (IMUSim).

```bash
source .venv/bin/activate

# Export 3D poses from video
python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_pose3d \
  --fps-target 20

# Synthesize virtual IMU signals from 3D poses
python -m pose_module.robot_emotions export-virtual-imu \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_virtual_imu \
  --fps-target 20
```

Outputs per clip under `output/<experiment>/<clip_id>/`:
- `pose/pose3d/pose3d.npz` — 3D skeleton trajectory
- `imu/virtual_imu.npz` — synthetic accelerometer + gyroscope

---

### Pipeline 2 — Text Prompt → Virtual IMU

Generates 3D motion from a natural-language description via T2M-GPT (VQ-VAE), then applies the same IMU synthesis chain.

```bash
source .venv/bin/activate

python -m pose_module.prompt_source.generate \
  --prompt "A person walks forward and waves" \
  --output-dir output/text2imu
```

Requires T2M-GPT weights in `pretrained/` and `checkpoints/t2m/`.

---

### Pipeline 3 — Window-Anchored Kimodo Generation

Uses real video windows as anchors: Qwen3-VL describes each 5-second segment; Kimodo generates new SMPL-X motion conditioned on the text and real joint positions.

**Step 1 — Export real pose3d** (same as Pipeline 1, `.venv`):

```bash
source .venv/bin/activate
python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions --domains 10ms 30ms \
  --output-dir output/robot_emotions_pose3d --fps-target 20
```

**Step 2 — Describe windows with Qwen3-VL** (`kimodo` env):

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows
```

**Step 3 — Build anchor catalog** (`kimodo` env):

```bash
conda activate kimodo
python -m robot_emotions_vlm build-anchor-catalog \
  --window-manifest-path output/robot_emotions_qwen_windows/window_description_manifest.jsonl \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors
```

**Step 4 — Generate with Kimodo** (`kimodo` env):

```bash
conda activate kimodo
python -m robot_emotions_vlm generate-kimodo \
  --model Kimodo-SMPLX-RP-v1 \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo
```

Each generated clip produces `motion.npz` + `motion_amass.npz` (SMPL-X) under `output/robot_emotions_kimodo/<clip_id>/`.

---

### Direct Kimodo generation (optional, `kimodo` env)

```bash
conda activate kimodo
kimodo_gen "A person sits down and stands up" \
  --model Kimodo-SMPLX-RP-v1 --duration 10.0 --output output/kimodo_direct/
```

## Citation

> Forneça o Bibtex para citação do repositório.
>
> Ele deve ser um bloco de código do tipo bibtex (\```bibtex CITAÇÃO \```), contendo uma citação do tipo ```@software```, para o repositório. Existe um script para gerar a citação automaticamente (veja ao final deste arquivo).
>
> A primeira citação deve ser ao código do repositório. Citação a outras produções relacionadas podem vir em seguida.

<!--Don't remove the following tags, it's used for placing the generated citation from the CFF file-->
<!--CITATION START-->
```bibtex
@software{

}
```
<!--CITATION END-->

## Authors

> Lista ordenada por data das pessoas que desenvolveram algo neste repositório. Deve ter ao menos 1 autor. Inclua todas as pessoas envolvidas no projeto e desenvolvimento
>
> Você também pode indicar um link para o perfil de algum autor: \[Nome do Autor]\(Link para o Perfil)
  
- (\<ano início>-\<ano fim>) \<Nome>: \<degree>, \<instituição>
  
## Acknowledgements

> Agradecimento as intituições de formento.

>Outros arquivos e informações que o repositório precisa ter:
> - Preencha a descrição do repositóio
>   - É necessário um _role_ de _admin_ no repositório para alterar sua descrição. Pessoas com [_role_ de _owner_](https://github.com/orgs/H-IAAC/people?query=role%3Aowner) na organização do GitHub podem alterar os papéis por repositório.
>   - Na página principal do repositório, na coluna direita, clique na engrenagem ao lado de "About"
>   - É recomendável também adicionar "topics" aos dados do repositório
> - Um arquivo LICENSE contendo a licença do repositório. Recomendamos a licença [LGPLv3](https://choosealicense.com/licenses/lgpl-3.0/).
>   - Converse com seu orientador caso acredite que essa licença não seja adequada. 
> - Um arquivo CFF contendo as informações sobre como citar o repositório.
>   - Este arquivo é lido automaticamente por ferramentas como o próprio GitHub ou o Zenodo, que geram automaticamente as citações.
>   - Existem ferramentas para auxiliar a criação do arquivo, como o [CFF Init](https://bit.ly/cffinit).    
>   - O script `generate_citation.py` pode ser utilizado para preencher o bloco de citação deste README automaticamente:
>     - ```bash
>         python -m pip install cffconvert
>         python generate_citation.py
>         ```
>   - Caso o arquivo tenha a tag `doi: <DOI>`, ele será lido automaticamente pelo Index.
> - Opcionalmente, o repositório pode ser preservado utilizando o Zenodo, que gerará um DOI para ele. [Tutorial](https://help.zenodo.org/docs/github/enable-repository/).
>   - É necessário um _role_ de _admin_ no repositório para publicar um repositório utilizando o Zenodo. Pessoas com [_role_ de _owner_](https://github.com/orgs/H-IAAC/people?query=role%3Aowner) na organização do GitHub podem alterar os papéis por repositório.