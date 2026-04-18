# Anchor Catalog

Documentacao curta do modulo `robot_emotions_vlm.anchor_catalog` e do fluxo window-level Qwen -> Kimodo.

## O que faz

O comando `build-anchor-catalog` cria um catalogo por janela a partir de:

- `pose3d_manifest.jsonl` exportado do ramo real
- `kimodo_window_prompt_catalog.jsonl` exportado pelo Qwen por janela

Para cada janela, ele:

- seleciona a janela temporal exata em `pose3d.npz`
- interpola a trajetoria 2D da raiz (`root_translation_m`) no grid de frames do Kimodo
- rebasa para que a primeira raiz comece em `x=z=0` e converte para o sistema de coordenadas do Kimodo (`-X`, `-Z`)
- escreve `constraints.json` somente com `root2d` denso
- salva `traceability.json`
- escreve uma entrada em `kimodo_anchor_catalog.jsonl`

## Contrato atual

Por padrao (`--hand-keyframes 0`), a unica constraint e `root2d`-only: ancora a trajetoria de chao da captura real e deixa o modelo gerar a pose livremente a partir do prompt textual.

- `root2d` denso (um ponto por frame do Kimodo), interpolado da trajetoria real
- janelas quase estaticas (`root2d_net_displacement_m < 0.05`) usam `root2d_motion_mode = stabilized_linear`
- `global_root_heading` e adicionado somente quando o deslocamento liquido justifica um heading confiavel (`>= 0.10 m`)
- nenhum retarget de pose e realizado; nenhuma dependencia do `kimodo` conda env no `build-anchor-catalog`

Com `--hand-keyframes N` (N > 0), adicionam-se constraints esparsas de end-effector para as maos:

- `left-hand` e `right-hand` com N keyframes uniformemente espacados na janela
- requer retarget IMUGPT22 → SMPLX22 (redimensiona comprimentos de osso preservando direcao)
- requer o ambiente conda `kimodo` (usa `_estimate_global_rotations_from_positions` do Kimodo)
- cada constraint inclui `local_joints_rot` (K, 22, 3) + `root_positions` (K, 3) para todos os 22 joints

Em uma frase: o catalogo ancora a trajetoria de chao da captura real enquanto o Kimodo gera pose natural a partir do prompt textual, com opcao de ancorar tambem as posicoes das maos.

## Defaults recomendados

Para manter boa cobertura temporal:

- `describe-windows --window-sec 5.0`
- `describe-windows --window-hop-sec 2.5`
- `describe-windows --num-video-frames 48`

Configuracao fixa desta versao:

- `root2d_min_displacement_m = 0.05`
- `heading_min_displacement_m = 0.10`

`build-anchor-catalog` usa `root2d`-only por padrao. Use `--hand-keyframes N` para adicionar end-effectors nas maos (requer env `kimodo`).

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

Modo padrao (`root2d` apenas):

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors_cc \
  --model Kimodo-SMPLX-RP-v1
```

Com end-effectors nas maos (requer env `kimodo`):

```bash
python -m robot_emotions_vlm build-anchor-catalog \
  --pose3d-manifest-path output/robot_emotions_pose3d/pose3d_manifest.jsonl \
  --qwen-window-catalog-path output/robot_emotions_qwen_windows/kimodo_window_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_anchors_hands \
  --model Kimodo-SMPLX-RP-v1 \
  --hand-keyframes 4
```

O `constraints.json` resultante salva:

- `root2d.frame_indices`, `root2d.smooth_root_2d`, `root2d.global_root_heading` quando aplicavel
- `left-hand` e `right-hand` com `local_joints_rot` (K, 22, 3) e `root_positions` (K, 3) quando `--hand-keyframes > 0`

### 4. Gerar motions com as ancoras

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors/kimodo_anchor_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo_generated
```

Para iterar em uma janela especifica:

```bash
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_kimodo_anchors_hands/kimodo_anchor_catalog.jsonl \
  --prompt-id robot_emotions_10ms_u02_tag11__w000 \
  --output-dir output/robot_emotions_kimodo_generated_single_hands
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
