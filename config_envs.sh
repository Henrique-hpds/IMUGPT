set -e

step() { echo; echo "==> $*"; }
ok()   { echo "    [ok] $*"; }

# 1. MMPose environment setup (3.8)
step "Creating openmmlab env (Python 3.8)..."
conda create --name openmmlab python=3.8 -y
ok "openmmlab env created"

# Do not change the specified versions of the OpenMMLab packages
# Also, do not change the order of installation
step "Installing PyTorch and OpenMMLab packages..."
conda install -n openmmlab pytorch torchvision -c pytorch -y
conda run -n openmmlab pip install -U pip setuptools wheel openmim
conda run -n openmmlab pip install fsspec
conda run -n openmmlab mim install "mmengine==0.10.5"
conda run -n openmmlab mim install "mmcv==2.1.0"
conda run -n openmmlab mim install "mmdet==3.2.0"
conda run -n openmmlab mim install "mmpose==1.3.2"
conda run -n openmmlab mim install "mmpretrain==1.2.0"
ok "openmmlab packages installed"

step "Download the local weights from the repository"
conda run -n openmmlab python -m pose_module.download_models --env-name openmmlab
ok "Local weights downloaded"

# 2. Pose_module environment setup (3.13.12)
step "Creating pose_module env (Python 3.13.12)..."
conda create --name pose_module python=3.13.12 -y
ok "pose_module env created"

step "Installing pose_module dependencies..."
conda run -n pose_module pip install -r pose_module/requirements.txt
ok "pose_module dependencies installed"

step "Downloading models into pose_module env..."
conda run -n pose_module python -m pose_module.download_models --env-name openmmlab
ok "Models downloaded"

# 3. Kimodo environment setup (3.10)
step "Creating kimodo env (Python 3.10)..."
conda create -n kimodo python=3.10 -y
ok "kimodo env created"

step "Installing Kimodo dependencies..."
conda install -n kimodo pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
# If torch import fails with an MKL error, install mkl 2023.1.0
conda install -n kimodo -y mkl=2023.1.0
cd kimodo
conda run -n kimodo pip install -e .
cd ..
conda run -n kimodo pip install --upgrade "transformers==5.1.0" huggingface_hub accelerate safetensors av
ok "Kimodo dependencies installed"

echo "==> All environments configured successfully."