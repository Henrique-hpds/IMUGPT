# Pose Module

Este modulo documenta como rodar o pipeline de pose implementado ate agora. No momento, o caminho integrado e o dataset `RobotEmotions`, cobrindo as etapas:

- `5.1 video_loader`
- `5.2 person_selector`
- `5.3 vitpose_2d_estimator`

## O que esta implementado

Fluxo atual:

- leitura do dataset `RobotEmotions`
- descoberta dos clipes e pares `CSV IMU + video`
- exportacao de `imu.npz` e `metadata.json`
- inferencia 2D com `ViTPose-B`
- associacao temporal e selecao da pessoa principal
- exportacao de `pose/pose2d.npz` e artefatos auxiliares

## Estrutura do modulo

- `pose_module/pipeline.py`: orquestracao das etapas 5.1 a 5.3
- `pose_module/interfaces.py`: contratos e estruturas canonicas
- `pose_module/io/video_loader.py`: metadata de video, fps e selecao de frames
- `pose_module/tracking/person_selector.py`: tracking simples e selecao do sujeito
- `pose_module/vitpose/estimator.py`: backend ViTPose no ambiente `openmmlab`
- `pose_module/vitpose/adapter.py`: conversao para `PoseSequence2D`
- `pose_module/processing/quality.py`: consolidacao de relatorios
- `pose_module/robot_emotions/extractor.py`: scanner e export especificos do dataset
- `pose_module/robot_emotions/pose2d.py`: wrapper do dataset sobre o pipeline generico
- `pose_module/robot_emotions/cli.py`: CLI atual

## Pre-requisitos

- estar na raiz do repositorio
- usar a `.venv` para os comandos Python do projeto
- ter o dataset em `data/RobotEmotions`
- ter `ffprobe` disponivel no sistema
- ter ao menos um interpretador Python com OpenMMLab configurado (`mmpose`, `mmdet`, `mmpretrain`), por exemplo o ambiente Conda `openmmlab`
  - `mmpose`
  - `mmdet`
  - `mmpretrain`

Checagem rapida:

```bash
.venv/bin/python -m pose_module.robot_emotions --help
conda run -n openmmlab python -c "import mmpose, mmdet, mmpretrain; print('openmmlab ok')"
ffprobe -version
```

## Como rodar

### 1. Listar os clipes do dataset

```bash
.venv/bin/python -m pose_module.robot_emotions \
  scan \
  --dataset-root data/RobotEmotions
```

### 2. Exportar apenas IMU e metadata

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-imu \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_extract
```

### 3. Exportar pose 2D ate a etapa 5.3

Esse comando usa o layout do extrator, garante `imu.npz` e `metadata.json` quando necessario, e salva os artefatos de pose em `pose/`.

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose2d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose2d \
  --env-name openmmlab

Selecao de ambiente do backend (`--env-name`):

- `auto` (padrao): tenta o Python atual e, se necessario, tenta o `openmmlab`
- `current`: usa somente o Python atual
- `<nome_do_env_conda>`: usa o Python do env Conda informado (exemplo: `openmmlab`)
```

Rodar sem gerar o video de debug:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose2d \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_pose2d \
  --clip-id robot_emotions_30ms_u06_tag18 \
  --no-debug
```

## Saidas geradas

Para cada clipe exportado:

- `imu.npz`
- `metadata.json`
- `pose/pose2d.npz`
- `pose/person_track.json`
- `pose/quality_report.json`
- `pose/backend_run.json`
- `pose/raw_predictions.json`
- `pose/debug_overlay.mp4` quando `save_debug=true`

Na raiz da exportacao:

- `manifest.jsonl`
- `summary.json`
- `pose_manifest.jsonl`
- `pose_summary.json`

Layout por clipe:

```text
<output_dir>/<domain>/user_<id>/<clip_id>/
<output_dir>/<domain>/user_<id>/<clip_id>/pose/
```

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

## Observacoes

- `fps_target` inicial recomendado: `20`
- se o video original tiver fps maior que o alvo, o pipeline faz decimacao temporal
- se o video original tiver fps menor que o alvo, o pipeline preserva o fps nativo nesta fase
- o backend atual usa `ViTPose-B`
- o fluxo padrao exporta todos os clipes encontrados nos dominios selecionados
- para restringir a execucao, use `--clip-id` ou `--domains`
- as proximas etapas do pipeline ainda nao foram implementadas aqui:
  - `pose2d_cleaner`
  - `motionbert_3d_lifter`
  - `skeleton_mapper`
  - `metric_normalizer`
  - `root_trajectory_estimator`
  - `ik_adapter`
  - `imusim_adapter`
