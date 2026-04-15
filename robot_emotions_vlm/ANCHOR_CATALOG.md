# Anchor Catalog

Documentacao curta do modulo `robot_emotions_vlm.anchor_catalog` e do fluxo window-level Qwen -> Kimodo.

## O que faz

O comando `build-anchor-catalog` cria um catalogo por janela a partir de:

- `pose3d_manifest.jsonl` exportado do ramo real
- `kimodo_window_prompt_catalog.jsonl` exportado pelo Qwen por janela

Para cada janela, ele:

- seleciona a janela temporal exata em `pose3d.npz`
- amostra keyframes esparsos diretamente da pose 3D pseudo-global original
- reordena as juntas para `SMPLX22`
- aplica grounding usando a altura de apoio estimada pelos tornozelos e pes
- escreve `constraints.json` com `fullbody` em `global_joints_positions`
- escreve `root2d` denso a partir da trajetoria da raiz
- salva `traceability.json`
- escreve uma entrada em `kimodo_anchor_catalog.jsonl`

## Contrato atual

Esta linha principal e totalmente window-level e centrada em fidelidade geometrica da pose amostrada:

- `fullbody` nao usa `ik_sequence.npz`
- o artefato canonico da pose vem de `pose3d.npz`
- o payload do `fullbody` salva `global_joints_positions` grounded como a fonte de verdade
- `root_positions` e `smooth_root_2d` sao derivados da mesma pose grounded
- `root2d` continua sendo emitido para estabilizar a trajetoria global
- janelas quase estaticas usam `root2d_motion_mode = stabilized_linear`
- `global_root_heading` so entra quando o deslocamento liquido da raiz justifica heading confiavel
- `generate-kimodo` desliga o post-processamento automaticamente quando detecta este `fullbody` em espaco de pose

Em uma frase: o catalogo agora representa poses reais amostradas da janela, e nao uma parametrizacao IK intermediaria.

## Defaults recomendados

Para manter boa cobertura temporal sem perder fidelidade:

- `describe-windows --window-sec 5.0`
- `describe-windows --window-hop-sec 2.5`
- `describe-windows --num-video-frames 48`
- `build-anchor-catalog --constraint-keyframes 8`

Thresholds fixos desta versao:

- `root2d_min_displacement_m = 0.05`
- `heading_min_displacement_m = 0.10`
- `max_fullbody_keyframe_gap_sec = 0.5`

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

O `constraints.json` resultante salva:

- `fullbody.frame_indices`
- `fullbody.global_joints_positions`
- `fullbody.root_positions`
- `fullbody.smooth_root_2d`
- `root2d.frame_indices`
- `root2d.smooth_root_2d`
- `root2d.global_root_heading` quando aplicavel

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
