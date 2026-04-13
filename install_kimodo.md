# Instalação do Kimodo

## 1. Obter o código

Se o repositório ainda não tiver o submódulo:

```bash
git submodule add git@github.com:nv-tlabs/kimodo.git
git submodule update --init --recursive
```

Se o submódulo já existir:

```bash
git submodule update --init --recursive
```

## 2. Criar o ambiente

```bash
conda create -n kimodo python=3.10
conda activate kimodo
pip install --upgrade pip
```

## 3. Instalar PyTorch

Opcao recomendada via conda:

```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 4. Corrigir incompatibilidade conhecida do MKL

Em alguns ambientes, o `torch` falha ao importar com o erro abaixo:

```text
ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

Se isso acontecer, faça downgrade do `mkl` para uma versao compativel:

```bash
conda install -n kimodo -y mkl=2023.1.0
```

## 5. Instalar o Kimodo

```bash
cd kimodo
pip install -e .
```

Se voce quiser usar a demo interativa, instale os extras:

```bash
pip install -e ".[demo]"
```

## 6. Testar se o ambiente ficou correto

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
kimodo_gen --help
kimodo_textencoder --help
```

## 7. Dependencias para o modulo `robot_emotions_vlm`

O modulo novo roda diretamente no mesmo ambiente `kimodo`, sem `.venv`, sem `pose_module` e sem subprocesso para outro Python.

Se alguma dependencia estiver faltando, instale ou atualize no proprio `kimodo`:

```bash
conda activate kimodo
pip install --upgrade "transformers>=5.1.0" huggingface_hub accelerate safetensors av
```

Cheque se o import do backend Qwen3-VL funciona:

```bash
conda activate kimodo
python -c "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; import av; print('qwen3_vl_import_ok')"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

Observacao: no primeiro uso do `Qwen/Qwen3-VL-8B-Instruct`, os pesos podem ser baixados do Hugging Face.

## 8. Executar o novo modulo

Fluxo principal:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_qwen
```

Exemplo processando apenas um clipe:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --clip-id robot_emotions_10ms_u02_tag11 \
  --output-dir output/robot_emotions_qwen_single
```

Se voce ja tiver os pesos em cache e quiser forcar uso apenas local:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_qwen \
  --local-files-only
```

Saidas principais geradas pelo modulo:

- `output/robot_emotions_qwen/video_description_manifest.jsonl`
- `output/robot_emotions_qwen/video_description_summary.json`
- `output/robot_emotions_qwen/kimodo_prompt_catalog.jsonl`

Cada captura tambem recebe sua propria pasta com:

- `description.json`
- `raw_response.txt`
- `prompt_context.json`
- `quality_report.json`

## 9. Gerar movimentos no Kimodo a partir do catalogo

Depois que o catalogo textual for gerado pelo Qwen:

```bash
conda activate kimodo
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_qwen/kimodo_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo
```

O comando acima agora usa `Kimodo-SMPLX-RP-v1` por padrao para produzir tambem um arquivo AMASS ao lado de cada NPZ gerado.

Exemplo para apenas um clipe:

```bash
conda activate kimodo
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_qwen/kimodo_prompt_catalog.jsonl \
  --clip-id robot_emotions_10ms_u02_tag11 \
  --output-dir output/robot_emotions_kimodo_single
```

Arquivos principais dessa etapa:

- `output/robot_emotions_kimodo/kimodo_generation_manifest.jsonl`
- `output/robot_emotions_kimodo/kimodo_generation_summary.json`

Cada captura gerada recebe:

- `prompt_entry.json`
- `generation_config.json`
- `motion.npz` ou pasta `motion/` para multiplas amostras
- `motion_amass.npz` ao lado de cada NPZ quando o modelo SMPL-X estiver em uso
- opcionalmente `motion.bvh` ou `motion.csv`, dependendo do modelo Kimodo
## 10. Teste rapido do modulo

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos --help
python -m robot_emotions_vlm generate-kimodo --help
python -m unittest tests.test_robot_emotions_vlm -v
```

## 11. Execucao local

Os comandos abaixo rodam os modelos localmente. O que pode acontecer na primeira execucao e apenas o download dos pesos do Hugging Face.

Servidor local do text encoder:

```bash
kimodo_textencoder
```

Geracao local via CLI:

```bash
kimodo_gen "A person sitting, telling a story" \
  --model Kimodo-SMPLX-RP-v1 \
  --duration 10.0 \
  --output output/kimodo/
```

Para forcar uso apenas de cache/arquivos locais:

```bash
export LOCAL_CACHE=true
export TEXT_ENCODER_MODE=local
```
