# Anchor Catalog

Documentacao curta do modulo `robot_emotions_vlm.anchor_catalog` e do fluxo com ancoras reais.

## O que faz

O comando `build-anchor-catalog` cria um catalogo por janela a partir de:

- `pose3d_manifest.jsonl` exportado do ramo real
- `kimodo_window_prompt_catalog.jsonl` exportado pelo Qwen por janela

Para cada janela, ele:

- seleciona a janela temporal
- deriva `constraints.json`
- salva `traceability.json`
- escreve uma entrada em `kimodo_anchor_catalog.jsonl`

## Estado atual

Esta linha principal agora e totalmente window-level:

- o Qwen analisa cada janela individualmente
- constraints do tipo `root2d`
- `duration_hint_sec` por janela
- `num_samples` por janela
- filtros por `clip_id`, `prompt_id` e `window_id` na geracao

Ainda nao faz:

- retarget `fullbody`
- constraints de `left-hand`, `right-hand`, `left-foot`, `right-foot`

## Como rodar

### 1. Exportar pose3d real

Na `.venv` do projeto:

```bash
./.venv/bin/python -m pose_module.robot_emotions export-pose3d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose3d \
  --env-name openmmlab \
  --motionbert-device cuda:0 \
  --no-debug
```

### 2. Gerar o catalogo textual do Qwen por janela

No ambiente `kimodo`:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows \
  --window-sec 8.0 \
  --window-hop-sec 4.0 
```

### 3. Construir o catalogo ancorado

```bash
conda activate kimodo
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors \
  --model Kimodo-SMPLX-RP-v1
```

## Gerar motions com as ancoras

```bash
conda activate kimodo
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_generated
```

Para iterar em uma janela especifica:

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --prompt-id robot_emotions_10ms_u02_tag11__w000 \
  --output-dir output/robot_emotions_kimodo_generated_single
```

## Principais saidas

Em `build-anchor-catalog`:

- `kimodo_anchor_catalog.jsonl`
- `kimodo_anchor_catalog.summary.json`
- `<prompt_id>/constraints.json`
- `<prompt_id>/traceability.json`

Em `describe-windows`:

- `window_description_manifest.jsonl`
- `window_description_summary.json`
- `kimodo_window_prompt_catalog.jsonl`
- `<prompt_id>/window.mp4`

Em `generate-kimodo`:

- `kimodo_generation_manifest.jsonl`
- `kimodo_generation_summary.json`
- `<prompt_id>/motion.npz` ou pasta `motion/` quando `num_samples > 1`
- `motion_amass.npz` para `Kimodo-SMPLX-RP-v1`
