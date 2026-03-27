# Pose Module

Este modulo documenta como rodar o pipeline de pose implementado ate agora. No momento, o caminho integrado e o dataset `RobotEmotions`, cobrindo as etapas:

- `5.1 video_loader`
- `5.2 person_selector`
- `5.3 vitpose_2d_estimator`
- `5.4 pose2d_cleaner`
- `5.5 motionbert_3d_lifter`
- `5.6 skeleton_mapper`
- `5.7 metric_normalizer`
- `5.8 root_trajectory_estimator`
- `5.9 ik_adapter`
- `5.10 imusim_adapter`

## O que esta implementado

Fluxo atual:

- leitura do dataset `RobotEmotions`
- descoberta dos clipes e pares `CSV IMU + video`
- exportacao de `imu.npz` e `metadata.json`
- inferencia 2D com `ViTPose-B`
- associacao temporal e selecao da pessoa principal
- limpeza temporal 2D, remocao de outliers e adaptacao para `motionbert17`
- lifting 3D por janelas deslizantes no contrato `MotionBERT`
- mapeamento estrito `MB17 -> IMUGPT22`
- normalizacao metrica local no referencial do corpo
- estimativa de `root_translation_m` e composicao pseudo-global para downstream
- adaptacao da pose pseudo-global para o contrato de IK, com BVH e sequencia local de rotacoes
- sintese de IMU virtual a partir do resultado de IK e de um `sensor_layout`
- exportacao de `pose/pose2d.npz`, `pose/pose3d.npz`, `pose/pose3d_metric_local.npz`, `pose/ik_sequence.npz`, `pose/virtual_imu.npz` e artefatos auxiliares

## Estrutura do modulo

- `pose_module/pipeline.py`: orquestracao das etapas 5.1 a 5.10
- `pose_module/interfaces.py`: contratos e estruturas canonicas
- `pose_module/model_registry.py`: resolucao dos modelos locais em `pose_module/checkpoints/`
- `pose_module/openmmlab_runtime.py`: selecao compartilhada do Python do env `openmmlab`
- `pose_module/download_models.py`: downloader reproduzivel dos pesos via `mim`
- `pose_module/io/video_loader.py`: metadata de video, fps e selecao de frames
- `pose_module/tracking/person_selector.py`: tracking simples e selecao do sujeito
- `pose_module/vitpose/estimator.py`: backend ViTPose no ambiente `openmmlab`
- `pose_module/vitpose/adapter.py`: conversao para `PoseSequence2D`
- `pose_module/processing/cleaner2d.py`: limpeza temporal e adaptacao para `MotionBERT`
- `pose_module/motionbert/adapter.py`: janelas deslizantes e contrato tensorial `[B, T, J, C]`
- `pose_module/motionbert/lifter.py`: lifting 3D temporal e exportacao de artefatos da etapa 5.5
- `pose_module/processing/skeleton_mapper.py`: expansao deterministica de `motionbert17` para `IMUGPT22`
- `pose_module/processing/metric_normalizer.py`: referencial corporal + escala antropometrica + suavizacao da etapa 5.7
- `pose_module/processing/root_estimator.py`: estimativa da trajetoria do root e composicao pseudo-global da etapa 5.8
- `pose_module/export/ik_adapter.py`: adaptacao da pose pseudo-global para rotacoes locais, offsets e BVH da etapa 5.9
- `pose_module/export/imusim_adapter.py`: sintese de IMU virtual a partir do contrato de IK e `sensor_layout` da etapa 5.10
- `pose_module/processing/temporal_filters.py`: interpolacao e suavizacao temporal
- `pose_module/processing/quality.py`: consolidacao de relatorios
- `pose_module/robot_emotions/extractor.py`: scanner e export especificos do dataset
- `pose_module/robot_emotions/pose2d.py`: wrapper do dataset sobre o pipeline generico
- `pose_module/robot_emotions/pose3d.py`: export do pipeline completo ate a etapa 5.8
- `pose_module/robot_emotions/virtual_imu.py`: export do pipeline completo ate a etapa 5.10
- `pose_module/robot_emotions/cli.py`: CLI atual
- `pose_module/configs/sensor_layout.yaml`: layout padrao dos sensores virtuais

## Pre-requisitos

- estar na raiz do repositorio
- usar a `.venv` para os comandos Python do projeto
- ter o dataset em `data/RobotEmotions`
- ter `ffprobe` disponivel no sistema
- ter o ambiente Conda `openmmlab` com Python 3.8 e OpenMMLab (`mmpose`, `mmdet`, `mmpretrain`)
- ter os pesos locais em `pose_module/checkpoints/`

## Passo a passo

### 1. Validar os ambientes Python

Use a `.venv` para os scripts do projeto e o env `openmmlab` para os backends OpenMMLab:

```bash
.venv/bin/python -V
conda run -n openmmlab python -V
conda run -n openmmlab python -c "import mmpose, mmdet, mmpretrain; print('openmmlab ok')"
ffprobe -version
```

### 2. Baixar os pesos locais do repositorio

Esse passo popula `pose_module/checkpoints/` com os modelos usados pelo pipeline.

```bash
.venv/bin/python -m pose_module.download_models --env-name openmmlab
find pose_module/checkpoints -maxdepth 1 -type f | sort
```

Os modelos esperados sao:

- `td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192`
- `rtmdet_m_8xb32-300e_coco`
- `motionbert_dstformer-ft-243frm_8xb32-120e_h36m`

### 3. Conferir a CLI disponivel

```bash
.venv/bin/python -m pose_module.robot_emotions --help
```

A CLI atual expõe:

- `scan`
- `export-imu`
- `export-pose2d`
- `export-pose3d`
- `export-virtual-imu`

### 4. Listar os clipes do dataset

```bash
.venv/bin/python -m pose_module.robot_emotions \
  scan \
  --dataset-root data/RobotEmotions
```

Se quiser restringir dominios:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  scan \
  --dataset-root data/RobotEmotions \
  --domains 10ms
```

### 5. Exportar apenas IMU e metadata

Esse passo prepara `imu.npz`, `metadata.json`, `manifest.jsonl` e `summary.json`.

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_extract
```

Para um unico clipe:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_extract \
  --clip-id robot_emotions_10ms_u02_tag05
```

### 6. Rodar o pipeline 2D pela CLI

Esse comando roda o fluxo ate a limpeza 2D, cobrindo as etapas `5.1` a `5.4`.
Ele garante `imu.npz` e `metadata.json` quando necessario e salva os artefatos de pose em `pose/`.

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose2d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose2d \
  --clip-id robot_emotions_10ms_u02_tag05 \
  --env-name openmmlab
```

Sem video de debug:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose2d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose2d \
  --clip-id robot_emotions_10ms_u02_tag05 \
  --env-name openmmlab \
  --no-debug
```

Retire `--clip-id` para rodar todos os clipes encontrados nos dominios selecionados.

Selecao de ambiente do backend (`--env-name`):

- `openmmlab` (padrao): usa explicitamente o env Conda com Python 3.8 e OpenMMLab
- `auto`: tenta `openmmlab` primeiro e depois o Python atual
- `current`: usa somente o Python atual
- `<nome_do_env_conda>`: usa somente o Python do env Conda informado

### 7. Rodar o pipeline 3D completo 5.1 a 5.8 pela CLI

O comando abaixo executa o fluxo completo com MotionBERT:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose3d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose3d \
  --clip-id robot_emotions_10ms_u02_tag05 \
  --env-name openmmlab
```

Flags uteis do `export-pose3d`:

- `--debug-2d` / `--no-debug-2d`: liga ou desliga os overlays 2D (`debug_overlay.mp4`, `debug_overlay_raw.mp4`, `debug_overlay_clean.mp4`)
- `--debug-3d` / `--no-debug-3d`: liga ou desliga os overlays side-by-side 3D (`debug_overlay_pose3d_raw.mp4`, `debug_overlay_pose3d_imugpt22.mp4`)
- `--no-debug`: desliga todos os videos de debug de uma vez
- `--motionbert-env-name <env>`: usa um env Conda especifico para o backend 3D
- `--motionbert-window-size <int>`: janela temporal solicitada
- `--motionbert-window-overlap <float>`: overlap entre janelas
- `--motionbert-device <device>`: `auto`, `cpu`, `cuda:0`, etc.
- `--no-motionbert-confidence`: remove o canal de confianca da entrada do MotionBERT
- `--allow-motionbert-fallback-backend`: permite cair para o backend heuristico se o MotionBERT real falhar

Para gerar apenas o debug 3D, sem os videos 2D:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose3d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose3d \
  --clip-id robot_emotions_10ms_u02_tag01 \
  --env-name openmmlab \
  --debug-2d \
  --debug-3d
```

Esse fluxo gera os artefatos 3D das etapas `5.5`, `5.6`, `5.7` e `5.8`, com `pose3d.npz` no contrato final `IMUGPT22` em `pseudo_global_metric`, `pose3d_metric_local.npz` preservando a saida local metrica da etapa 5.7, `pose3d_motionbert17.npz` preservando a saida MB17 bruta do MotionBERT, `3d_keypoints_raw.npy`, `3d_keypoints_metric.npy`, `root_translation.npy`, `motionbert_run.json` e, quando habilitado, um debug side-by-side com 2D clean + 3D raw/final.

### 8. Rodar o pipeline completo 5.1 a 5.10 pela CLI

O comando abaixo fecha o pipeline com `IK` e `IMUSim`, preservando tambem os artefatos 3D intermediarios:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-virtual-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_virtual_imu \
  --clip-id robot_emotions_10ms_u02_tag01 \
  --env-name openmmlab \
  --no-debug-2d --no-debug-3d
```  

.venv/bin/python -m pose_module.robot_emotions export-virtual-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_virtual_imu_cal \
  --clip-id robot_emotions_10ms_u02_tag01 \
  --fps-target 20 \
  --real-imu-label-key action \
  --real-imu-signal-mode both \
  --real-imu-percentile-resolution 100 \
  --no-debug


Flags uteis do `export-virtual-imu`:

- `--sensor-layout-path <arquivo>`: usa um layout customizado de sensores virtuais
- `--imu-acc-noise-std-m-s2 <float>`: injeta ruido gaussiano no acelerometro
- `--imu-gyro-noise-std-rad-s <float>`: injeta ruido gaussiano no giroscopio
- `--imu-random-seed <int>`: fixa a seed do ruido opcional
- `--real-imu-reference-path <arquivo.npz>`: aplica a calibracao virtual->real por percentis usando uma referencia real
- `--real-imu-label-key <campo>`: usa um campo de `labels` do manifest (`action`, `emotion`, `stimulus`) para calibracao por classe
- `--real-imu-signal-mode <acc|gyro|both>`: escolhe quais canais serao calibrados contra a referencia real
- `--real-imu-percentile-resolution <int>`: controla o numero de percentis usados no mapeamento do artigo
- `--no-real-imu-per-class-calibration`: desabilita a calibracao por classe e usa a distribuicao global da referencia
- `--debug-2d` / `--no-debug-2d`: controla os overlays 2D do trecho 3D do pipeline
- `--debug-3d` / `--no-debug-3d`: controla os overlays 3D do trecho 3D do pipeline

Esse fluxo gera os artefatos finais das etapas `5.9` e `5.10`, incluindo `ik_sequence.npz`, `pose3d_ik.bvh`, `virtual_imu.npz`, `virtual_imu_report.json`, `sensor_layout_resolved.json` e um `quality_report.json` consolidado com pose 3D, IK e IMU virtual. Quando a referencia real e fornecida, o pipeline tambem salva `virtual_imu_raw.npz` e `virtual_imu_calibration_report.json`.

O `sensor_layout` padrao fica em `pose_module/configs/sensor_layout.yaml` e replica o arranjo de 4 sensores do `RobotEmotions` (`waist`, `head`, `right_forearm`, `left_forearm`). Se o seu downstream exigir outro arranjo, troque esse arquivo pelo layout correspondente.

### 9. Exportar BVH pela CLI (arquivo customizado)

Voce pode exportar qualquer `pose3d.npz` (incluindo esqueletos diferentes, como `motionbert17`) para BVH com caminho de saida customizado:

```bash
.venv/bin/python -m pose_module.export.bvh \
  --pose3d-npz output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/pose3d_motionbert17.npz \
  --output-bvh output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/meu_pipeline_3d_raw.bvh
```

Argumentos da CLI:

- `--pose3d-npz`: caminho do arquivo de entrada no formato `PoseSequence3D` (`.npz`)
- `--output-bvh`: caminho do arquivo BVH de saida
- `--no-ground-to-floor`: opcional, desliga o ajuste vertical que coloca o menor ponto no chao

Exemplo para exportar o esqueleto final do pipeline (`pose3d.npz`):

```bash
.venv/bin/python -m pose_module.export.bvh \
  --pose3d-npz output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/pose3d.npz \
  --output-bvh output/robot_emotions_pose3d/10ms/user_02/robot_emotions_10ms_u02_tag05/pose/pose3d_final.bvh
```

## Saidas geradas

### Saidas do `export-imu`

Na raiz da exportacao:

- `manifest.jsonl`
- `summary.json`

Para cada clipe exportado:

- `imu.npz`
- `metadata.json`

Layout por clipe:

```text
<output_dir>/<domain>/user_<id>/<clip_id>/
```

### Saidas do `export-pose2d`

Na raiz da exportacao:

- `pose_manifest.jsonl`
- `pose_summary.json`

Para cada clipe exportado:

- `pose/pose2d.npz`
- `pose/2d_keypoints_raw.npy`
- `pose/2d_keypoints_clean.npy`
- `pose/person_track.json`
- `pose/quality_report.json`
- `pose/backend_run.json`
- `pose/raw_predictions.json`
- `pose/debug_overlay.mp4` quando `save_debug=true`
- `pose/debug_overlay_raw.mp4` quando `save_debug=true`
- `pose/debug_overlay_clean.mp4` quando `save_debug=true`

Layout por clipe:

```text
<output_dir>/<domain>/user_<id>/<clip_id>/pose/
```

### Saidas do `export-pose3d`

Na raiz da exportacao:

- `pose3d_manifest.jsonl`
- `pose3d_summary.json`

Quando voce roda o pipeline completo ate a etapa `5.8`, os arquivos abaixo sao adicionados ao diretorio `pose/`:

- `pose/pose3d.npz`
- `pose/pose3d_metric_local.npz`
- `pose/pose3d_motionbert17.npz`
- `pose/pose3d.bvh`
- `pose/3d_keypoints_raw.npy`
- `pose/3d_keypoints_metric.npy`
- `pose/root_translation.npy`
- `pose/motionbert_run.json`
- `pose/debug_overlay_pose3d_raw.mp4` quando `save_debug=true` ou `save_debug_3d=true`
- `pose/debug_overlay_pose3d_imugpt22.mp4` quando `save_debug=true` ou `save_debug_3d=true`

### Saidas do `export-virtual-imu`

Na raiz da exportacao:

- `virtual_imu_manifest.jsonl`
- `virtual_imu_summary.json`

Quando voce roda o pipeline completo ate a etapa `5.10`, os arquivos abaixo sao adicionados ao diretorio `pose/`:

- todos os artefatos do `export-pose3d`
- `pose/ik_sequence.npz`
- `pose/ik_report.json`
- `pose/pose3d_ik.bvh`
- `pose/virtual_imu.npz`
- `pose/virtual_imu_report.json`
- `pose/sensor_layout_resolved.json`
- `pose/quality_report.json`

No pipeline 3D, os overlays 2D e 3D podem ser controlados separadamente com `save_debug_2d` e `save_debug_3d`.

Se uma mesma `Tag` tiver mais de uma captura, o extrator gera um registro por captura, por exemplo `...tag07` e `...tag07_2`.

## Formato principal dos artefatos

No `imu.npz`:

- `imu`: `np.ndarray[T, 4, 6]`
- `imu_flat`: `np.ndarray[T, 24]`
- `timestamps_sec`: `np.ndarray[T]`

No `pose/pose2d.npz`:

- `keypoints_xy`: `np.ndarray[T, 17, 2]`
- `confidence`: `np.ndarray[T, 17]`
- `bbox_xywh`: `np.ndarray[T, 4]`
- `frame_indices`: `np.ndarray[T]`
- `timestamps_sec`: `np.ndarray[T]`

Nos artefatos auxiliares da etapa 5.4:

- `2d_keypoints_raw.npy`: `np.ndarray[T, 17, 2]` na saida canonica do ViTPose
- `2d_keypoints_clean.npy`: `np.ndarray[T, 17, 2]` ja limpa e normalizada para o contrato `motionbert17`

No `pose/pose3d.npz`:

- `joint_positions_xyz`: `np.ndarray[T, 22, 3]`
- `joint_confidence`: `np.ndarray[T, 22]`
- `skeleton_parents`: `np.ndarray[22]`
- `root_translation_m`: `np.ndarray[T, 3]`
- `coordinate_space`: `pseudo_global_metric`

No `pose/pose3d_metric_local.npz`:

- `joint_positions_xyz`: `np.ndarray[T, 22, 3]`
- `joint_confidence`: `np.ndarray[T, 22]`
- `skeleton_parents`: `np.ndarray[22]`
- `coordinate_space`: `body_metric_local`

No `pose/pose3d_motionbert17.npz`:

- `joint_positions_xyz`: `np.ndarray[T, 17, 3]`
- `joint_confidence`: `np.ndarray[T, 17]`
- `skeleton_parents`: `np.ndarray[17]`

No `pose/ik_sequence.npz`:

- `local_joint_rotations`: `np.ndarray[T, 22, 4]` em quaternions locais no contrato do IK
- `root_translation_m`: `np.ndarray[T, 3]`
- `joint_offsets_m`: `np.ndarray[22, 3]`
- `skeleton_parents`: `np.ndarray[22]`
- `joint_names_3d`: `np.ndarray[22]`

No `pose/virtual_imu.npz`:

- `acc`: `np.ndarray[T, S, 3]`
- `gyro`: `np.ndarray[T, S, 3]`
- `sensor_names`: `np.ndarray[S]`
- `timestamps_sec`: `np.ndarray[T]`

Nos artefatos auxiliares das etapas 5.5, 5.6, 5.7, 5.8, 5.9 e 5.10:

- `3d_keypoints_raw.npy`: `np.ndarray[T, 17, 3]` em referencial de camera
- `3d_keypoints_metric.npy`: `np.ndarray[T, 22, 3]` apos normalizacao metrica local e suavizacao temporal
- `root_translation.npy`: `np.ndarray[T, 3]` com a trajetoria pseudo-global estimada do pelvis/root
- `motionbert_run.json`: resumo do backend, janelas e qualidade do lifting 3D
- `ik_report.json`: resumo da adaptacao para IK, incluindo erro medio de reconstrucao
- `pose3d_ik.bvh`: export BVH derivado da saida pseudo-global final
- `virtual_imu_report.json`: resumo da sintese de IMU virtual e faixas dinamicas dos sensores
- `virtual_imu_raw.npz`: IMU virtual bruto antes da calibracao contra IMU real, quando a calibracao opcional esta ativa
- `virtual_imu_calibration_report.json`: resumo da calibracao por percentis com a referencia real, quando ativa
- `sensor_layout_resolved.json`: layout de sensores efetivamente resolvido contra o esqueleto IMUGPT22
- `quality_report.json`: consolidacao final da qualidade do clip, incluindo pose 3D, IK e IMU virtual
- `debug_overlay_pose3d_raw.mp4`: video lado a lado com o video original + pose 2D clean e a pose 3D raw do MotionBERT
- `debug_overlay_pose3d_imugpt22.mp4`: video lado a lado com o video original + pose 2D clean e a pose 3D local metrica no esqueleto IMUGPT22 apos a etapa 5.7

No comando `export-pose3d`, voce pode controlar os overlays separadamente com `--debug-2d` / `--no-debug-2d` e `--debug-3d` / `--no-debug-3d`.

## Observacoes

- `fps_target` inicial recomendado: `20`
- se o video original tiver fps maior que o alvo, o pipeline faz decimacao temporal
- se o video original tiver fps menor que o alvo, o pipeline preserva o fps nativo nesta fase
- o backend atual usa `ViTPose-B`
- a etapa 5.5 agora tenta primeiro um backend MotionBERT real via `mmpose` no env `openmmlab`
- os pesos usados pelo pipeline ficam explicitamente em `pose_module/checkpoints/`
- para popular essa pasta em outra maquina, rode `.venv/bin/python -m pose_module.download_models --env-name openmmlab`
- o ViTPose 2D usa `td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192`
- o detector de pessoa usa `rtmdet_m_8xb32-300e_coco`
- o MotionBERT 3D usa `motionbert_dstformer-ft-243frm_8xb32-120e_h36m`
- o fallback heuristico existe apenas como opcional e fica desabilitado por padrao
- o fluxo padrao exporta todos os clipes encontrados nos dominios selecionados
- para restringir a execucao, use `--clip-id` ou `--domains`
- o `export-pose3d` termina na etapa `5.8`, preservando a pose pseudo-global final
- o `export-virtual-imu` fecha o pipeline ate `5.10`, reutilizando a mesma saida 3D final
