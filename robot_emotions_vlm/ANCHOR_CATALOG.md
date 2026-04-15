# Anchor Catalog

Documentacao curta do modulo `robot_emotions_vlm.anchor_catalog` e do fluxo window-level Qwen -> Kimodo.

## O que faz

O comando `build-anchor-catalog` cria um catalogo por janela a partir de:

- `pose3d_manifest.jsonl` exportado do ramo real
- `kimodo_window_prompt_catalog.jsonl` exportado pelo Qwen por janela

Para cada janela, ele:

- seleciona a janela temporal exata
- reutiliza `pose/ik_sequence.npz` quando ele ja existe
- materializa `pose/ik_sequence.npz` ao lado de `pose3d.npz` quando o export real ainda so tem pose3d
- gera `constraints.json` com `fullbody` esparso e `root2d` condicional
- salva `traceability.json`
- escreve uma entrada em `kimodo_anchor_catalog.jsonl`

## Estado atual

Esta linha principal agora e totalmente window-level e orientada para fidelidade no alvo `Kimodo-SMPLX-RP-v1`:

- o Qwen analisa cada janela individualmente
- `prompt_text` passa por sanitizacao motion-only para ficar mais simples e compatível com o Kimodo
- `fullbody` sempre e emitido com keyframes esparsos por janela
- `root2d` so e emitido quando o deslocamento liquido da raiz na janela e relevante
- `global_root_heading` so entra quando o deslocamento liquido da raiz justifica orientacao confiavel
- `duration_hint_sec` e preservado por janela
- `num_samples` continua por janela
- filtros por `clip_id`, `prompt_id` e `window_id` continuam suportados na geracao

Nesta correcao, o modo rico e suportado apenas para alvo `SMPLX`.

## Defaults recomendados

Para reduzir achatamento semantico e melhorar coerencia temporal:

- `describe-windows --window-sec 5.0`
- `describe-windows --window-hop-sec 2.5`
- `describe-windows --num-video-frames 48`
- `build-anchor-catalog --constraint-keyframes 8`

Thresholds fixos desta versao:

- `root2d_min_displacement_m = 0.05`
- `heading_min_displacement_m = 0.10`

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

Se `pose/ik_sequence.npz` ja existir ao lado de `pose/pose3d.npz`, ele sera reutilizado.
Se ainda nao existir, `build-anchor-catalog` agora o gera automaticamente a partir do `pose3d.npz` e o salva em `pose/ik_sequence.npz`.

### 2. Gerar o catalogo textual do Qwen por janela

No ambiente `kimodo`:

```bash
python -m robot_emotions_vlm describe-windows \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --output-dir output/robot_emotions_qwen_windows \
  --window-sec 5.0 \
  --window-hop-sec 2.5 \
  --num-video-frames 48
```

### 3. Construir o catalogo ancorado

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors \
  --model Kimodo-SMPLX-RP-v1 \
  --constraint-keyframes 8
```

### 4. Gerar motions com as ancoras

```bash
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
