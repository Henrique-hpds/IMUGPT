# MMPose instalation via Conda

Pose preprocessing stack for pose -> IMU workflows.

## Installation (Conda, Recommended)

First, make sure Miniconda is installed. If not, follow the [official instructions here](https://www.anaconda.com/docs/getting-started/miniconda/main).

Run from repository root (`Pose2IMU/`).

### 1) Create and activate environment

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### 2) Install PyTorch + base tooling

```bash
conda install pytorch torchvision -c pytorch
pip install -U pip setuptools wheel openmim
pip install fsspec
```

### 3) Install OpenMMLab stack (fixed versions)

```bash
mim install "mmengine==0.10.5"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"
mim install "mmpretrain==1.2.0"
```

### 4) Download checkpoints used by this pipeline

Recommended (downloads the exact set required by `pose_module`):

```bash
.venv/bin/python -m pose_module.download_models --env-name openmmlab
```

Equivalent manual commands:

```bash
conda run -n openmmlab python -m mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest pose_module/checkpoints
conda run -n openmmlab python -m mim download mmpose --config td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192 --dest pose_module/checkpoints
conda run -n openmmlab python -m mim download mmpose --config motionbert_dstformer-ft-243frm_8xb32-120e_h36m --dest pose_module/checkpoints
```
