set -e

step() { echo; echo "==> $*"; }
ok()   { echo "    [ok] $*"; }

# 1. Pose_module environment setup (3.13.12)
step "Creating pose_module env (Python 3.13.12)..."
conda create --name pose_module python=3.13.12 -y
ok "pose_module env created"

step "Installing pose_module dependencies..."
conda activate pose_module
pip install -r pose_module/requirements.txt
conda deactivate
ok "pose_module dependencies installed"

step "Downloading models into pose_module env..."
conda run -n pose_module python -m pose_module.download_models --env-name openmmlab
ok "Models downloaded"

# 2. MMPose environment setup (3.8)
step "Creating openmmlab env (Python 3.8)..."
conda create --name openmmlab python=3.8 -y
ok "openmmlab env created"

step "Installing PyTorch and OpenMMLab packages..."
conda activate openmmlab
conda install pytorch torchvision -c pytorch
pip install -U pip setuptools wheel openmim
pip install fsspec
mim install "mmengine==0.10.5"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"
mim install "mmpretrain==1.2.0"
conda deactivate
ok "openmmlab packages installed"

# 3. Kimodo environment setup (3.10)
step "Creating kimodo env (Python 3.10)..."
conda create -n kimodo python=3.10 -y
ok "kimodo env created"

step "Installing Kimodo dependencies..."
conda activate kimodo
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
# If torch import fails with an MKL error:
# conda install -n kimodo -y mkl=2023.1.0
cd kimodo && pip install -e . && cd ..
pip install --upgrade "transformers>=5.1.0" huggingface_hub accelerate safetensors av
conda deactivate
ok "Kimodo dependencies installed"

echo
echo "==> All environments configured successfully."