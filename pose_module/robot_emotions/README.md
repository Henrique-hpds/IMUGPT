# RobotEmotions Extractor

Esta etapa prepara o dataset `RobotEmotions` para uso futuro no pipeline de pose, sem implementar o pipeline em si.

## O que faz

- varre `data/RobotEmotions/{10ms,30ms}`
- localiza pares `CSV IMU + vídeo`
- preserva múltiplas capturas na mesma `Tag` quando existirem arquivos como `_2`
- lê e limpa os timestamps
- extrai IMUs no formato `imu[T, 4, 6]`
- aplica o mapeamento:
  - `1 = waist`
  - `2 = head`
  - `3 = left_forearm`
  - `4 = right_forearm`
- gera artefatos prontos para consumo posterior

## Saídas

Para cada clipe exportado:

- `imu.npz`
- `metadata.json`

Na raiz da exportação:

- `manifest.jsonl`
- `summary.json`

Observação:

- se uma mesma `Tag` tiver mais de uma captura, o extrator gera um registro por captura, por exemplo `...tag07` e `...tag07_2`

## Como rodar

Scan simples:

```bash
.venv/bin/python -m pose_module.robot_emotions --dataset-root data/RobotEmotions
```

Exportar artefatos:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  --dataset-root data/RobotEmotions \
  --output-dir /tmp/robot_emotions_extract
```

Exportar só alguns clipes para teste:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  --dataset-root data/RobotEmotions \
  --output-dir /tmp/robot_emotions_extract \
  --limit 5
```

## Formato principal

O dado principal salvo é:

- `imu`: `np.ndarray[T, 4, 6]`
- `imu_flat`: `np.ndarray[T, 24]`
- `timestamps_sec`: `np.ndarray[T]`
