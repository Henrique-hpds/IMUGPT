[file name]: pipeline.md
[file content begin]
A seguir está um plano de integração completo para substituir o módulo de geração de pose do IMUGPT original por um pipeline ViTPose → MotionBERT, preservando o restante do fluxo IK → IMUSim → calibração → HAR. No artigo/repositório original, o IMUGPT usa texto → T2M-GPT → sequência 3D de 22 juntas → IK → IMUSim. O IK recebe posições 3D + hierarquia do esqueleto, estima rotações locais e translação global do root, e o IMUSim usa isso para sintetizar aceleração e velocidade angular. O repositório público do projeto também descreve explicitamente esse pipeline.

## 1. Objetivo do módulo

Substituir apenas a etapa:

- antes: texto → T2M-GPT → poses 3D
- depois: vídeo → ViTPose → keypoints 2D → MotionBERT → poses 3D

Mantém-se inalterado:

- IK
- IMUSim
- calibração domínio virtual→real
- treinamento/classificação HAR

A decisão de usar ViTPose é adequada porque ele é um estimador oficial de pose 2D baseado em Vision Transformer. A decisão de usar MotionBERT também é adequada porque o repositório oficial inclui 3D pose estimation a partir de sequências de skeletons 2D, com entrada tensorial do tipo [batch, frames, joints, channels].

## 2. Premissas de projeto

Este módulo deve assumir, como contrato operacional:

- vídeos monoculares
- um único sujeito ativo por clipe
- câmera preferencialmente fixa ou com pouco egomotion
- atividades locomotoras e ADLs simples
- processamento offline inicialmente; streaming em tempo real fica para fase 2

A restrição de single person é coerente com a documentação de inferência “in the wild” do MotionBERT, que explicita suporte atual a uma pessoa por vídeo.

## 3. Arquitetura-alvo

### 3.1 Fluxo lógico
- video_loader
- person_selector
- vitpose_2d_estimator
- pose2d_cleaner
- motionbert_3d_lifter
- skeleton_mapper
- metric_normalizer
- root_trajectory_estimator
- ik_adapter
- imusim_adapter
- existing_calibration_and_har
### 3.2 Ponto de integração no IMUGPT

O ponto de substituição deve ser a interface atualmente equivalente a:

- “gerar sequência de poses 3D”
saída esperada pelo restante do código:
- sequência temporal de juntas 3D
- hierarquia do esqueleto
- taxa de amostragem

No pipeline original, o T2M-GPT gera uma sequência de movimento 3D, depois o IK converte isso para rotações locais e root translation, e então o IMUSim produz IMU virtual.

## 4. Especificação funcional do módulo de pose

### 4.1 Entrada

O módulo deve aceitar:

- video_path: str
- activity_label: Optional[str]
- subject_id: Optional[str]
- fps_target: int
- camera_meta: Optional[dict]
- clip_id: str
Formato esperado do vídeo
- codec comum (mp4, mov, avi)
- resolução mínima recomendada: 720p
- fps original entre 24 e 60
- sujeito ocupando pelo menos ~20% da altura do frame
### 4.2 Saída principal

O módulo deve exportar um objeto unificado:

PoseSequence3D = {
    "clip_id": str,
    "fps": int,
    "num_frames": int,
    "joint_names": List[str],
    "joint_positions_global_m": np.ndarray,   # [T, J, 3]
    "joint_confidence": np.ndarray,           # [T, J]
    "root_translation_m": np.ndarray,         # [T, 3]
    "skeleton_parents": List[int],
    "source": "vitpose_motionbert",
    "metadata": dict
}

Essa saída deve ser suficiente para o adaptador de IK.

### 4.3 Saída intermediária opcional

Salvar também:

- 2d_keypoints_raw.npy
- 2d_keypoints_clean.npy
- 3d_keypoints_raw.npy
- 3d_keypoints_metric.npy
- debug_overlay.mp4
- quality_report.json

Isso reduz custo de depuração.

## 5. Submódulos e especificações

### 5.1 video_loader

Responsabilidades:

- abrir vídeo
- extrair frames
- normalizar fps para fps_target
- manter timestamps reais
Requisito

Internamente, o pipeline deve operar com um fps_target único. Para compatibilidade com o artigo, o alvo inicial recomendado é 20 Hz, já que os datasets reais foram reduzidos a 20 Hz para combinar com o virtual.

Política
- se fps_original > fps_target: decimar com preservação temporal
- se fps_original < fps_target: não interpolar no estágio inicial; manter fps nativo e só reamostrar depois do 3D
### 5.2 person_selector

Responsabilidades:

- selecionar a pessoa principal
- manter identidade temporal
- rejeitar clipes com troca de sujeito
Estratégia

Como o usuário quer ViTPose + MotionBERT, o plano mais robusto é:

- detector de pessoa por frame
- associação temporal simples por IoU + centroide
- selecionar o track de maior duração
Saída
- bounding box por frame
- track_id
- score de estabilidade do track
### 5.3 vitpose_2d_estimator

Responsabilidades:

- inferir pose 2D por frame sobre o bounding box selecionado
- retornar keypoints e confidence scores

ViTPose é um repositório oficial para estimação de pose humana baseado em Vision Transformer.

Saída canônica
PoseSequence2D = {
    "fps": int,
    "num_frames": int,
    "joint_names_2d": List[str],
    "keypoints_xy": np.ndarray,      # [T, J2d, 2]
    "confidence": np.ndarray,        # [T, J2d]
    "bbox_xywh": np.ndarray,         # [T, 4]
}
Regras
- processar 1 pessoa por frame
- guardar score por junta
- marcar juntas ausentes com NaN + confidence zero
Configuração recomendada
- começar com modelo ViTPose-B
- inferência em batch quando memória permitir
- backbone congelado; sem fine-tuning na fase 1
### 5.4 pose2d_cleaner

Responsabilidades:

- preencher pequenas lacunas
- remover jitter
- eliminar outliers
- converter o formato de juntas do ViTPose para o formato esperado pelo MotionBERT

Isso é necessário porque o MotionBERT trabalha com skeletons 2D sequenciais e a entrada de aplicação descrita no repositório é tensorial por frames e juntas.

Operações mínimas
- interpolação temporal curta para gaps até 5 frames
- filtro Savitzky–Golay ou low-pass temporal
- clipping de velocidade angular impossível entre frames
- normalização espacial por bbox ou pelvis-centered
Critérios de descarte
- mais de 20% das juntas ausentes em mais de 30% dos frames
- descontinuidade do track principal
- bbox colapsada ou sujeito fora de cena
### 5.5 motionbert_3d_lifter

Responsabilidades:

- converter a sequência 2D limpa para pose 3D temporal

O MotionBERT é o repositório oficial do paper ICCV 2023 e documenta uso para 3D pose estimation, com entrada sequencial de skeletons 2D.

Entrada
x.shape == [B, T, J, C]

com C contendo pelo menos coordenadas 2D; opcionalmente confidence pode ser incluída em canal adicional no adaptador.

Saída
keypoints_3d_camera = np.ndarray  # [T, J3d, 3]
Janela temporal
- usar inferência por janelas deslizantes
- window = 81 frames inicial
- overlap de 50%
Observação crítica

A documentação pública do MotionBERT para “in-the-wild” menciona AlphaPose para gerar 2D. Aqui a substituição por ViTPose é uma decisão de engenharia, não uma exigência do modelo; o módulo adapter deve apenas reproduzir o formato de skeleton 2D esperado.

### 5.6 skeleton_mapper

Responsabilidades:

- mapear o esqueleto 2D/3D do ViTPose/MotionBERT para o esqueleto do IMUGPT/IK
- definir pais de cada junta
- padronizar nomes

No artigo, o decoder do T2M-GPT produz sequência de 22 juntas. Portanto, o módulo novo deve exportar um conjunto de juntas compatível com o IK já usado no IMUGPT.

Recomendação

Congelar o contrato de esqueleto em duas etapas: (1) normalizar a saída do MotionBERT para um esqueleto intermediário fixo de 17 juntas e (2) expandir para o esqueleto IK de 22 juntas já usado pelo projeto.

#### 5.6.1 Esqueleto intermediário fixo (MB17)

O `motionbert/adapter.py` deve sempre emitir exatamente esta ordem de juntas (índice fixo):

| idx | joint_mb17 |
|---:|---|
| 0 | pelvis |
| 1 | left_hip |
| 2 | right_hip |
| 3 | spine |
| 4 | left_knee |
| 5 | right_knee |
| 6 | thorax |
| 7 | left_ankle |
| 8 | right_ankle |
| 9 | neck |
| 10 | head |
| 11 | left_shoulder |
| 12 | right_shoulder |
| 13 | left_elbow |
| 14 | right_elbow |
| 15 | left_wrist |
| 16 | right_wrist |

Se o checkpoint do MotionBERT usado não produzir essa ordem nativamente, a reordenação ocorre no adapter, nunca no `skeleton_mapper`.

#### 5.6.2 Esqueleto-alvo congelado (IMUGPT22)

O `skeleton_mapper` deve exportar exatamente o esqueleto abaixo (ordem e pais fixos), consistente com a árvore já usada no projeto:

| idx | joint_imugpt22 | parent_idx | parent_name |
|---:|---|---:|---|
| 0 | Pelvis | -1 | ROOT |
| 1 | Left_hip | 0 | Pelvis |
| 2 | Right_hip | 0 | Pelvis |
| 3 | Spine1 | 0 | Pelvis |
| 4 | Left_knee | 1 | Left_hip |
| 5 | Right_knee | 2 | Right_hip |
| 6 | Spine2 | 3 | Spine1 |
| 7 | Left_ankle | 4 | Left_knee |
| 8 | Right_ankle | 5 | Right_knee |
| 9 | Spine3 | 6 | Spine2 |
| 10 | Left_foot | 7 | Left_ankle |
| 11 | Right_foot | 8 | Right_ankle |
| 12 | Neck | 9 | Spine3 |
| 13 | Left_collar | 9 | Spine3 |
| 14 | Right_collar | 9 | Spine3 |
| 15 | Head | 12 | Neck |
| 16 | Left_shoulder | 13 | Left_collar |
| 17 | Right_shoulder | 14 | Right_collar |
| 18 | Left_elbow | 16 | Left_shoulder |
| 19 | Right_elbow | 17 | Right_shoulder |
| 20 | Left_wrist | 18 | Left_elbow |
| 21 | Right_wrist | 19 | Right_elbow |

`parents` final obrigatório:
[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

#### 5.6.3 Regras de mapeamento MB17 -> IMUGPT22

Regras determinísticas (sem heurística aberta):

1. Cópia direta
- `Pelvis <- pelvis`
- `Left_hip <- left_hip`, `Right_hip <- right_hip`
- `Left_knee <- left_knee`, `Right_knee <- right_knee`
- `Left_ankle <- left_ankle`, `Right_ankle <- right_ankle`
- `Neck <- neck`, `Head <- head`
- `Left_shoulder <- left_shoulder`, `Right_shoulder <- right_shoulder`
- `Left_elbow <- left_elbow`, `Right_elbow <- right_elbow`
- `Left_wrist <- left_wrist`, `Right_wrist <- right_wrist`

2. Coluna vertebral (3 segmentos)
- `Spine1 = Pelvis + (1/3) * (thorax - pelvis)`
- `Spine2 = Pelvis + (2/3) * (thorax - pelvis)`
- `Spine3 = thorax`

3. Clavículas
- `Left_collar = 0.5 * (Spine3 + Left_shoulder)`
- `Right_collar = 0.5 * (Spine3 + Right_shoulder)`

4. Pés (juntas ausentes no MB17)
- Construir eixos do corpo por frame:
  - `v_right = normalize(Right_hip - Left_hip)`
  - `v_up = normalize(Neck - Pelvis)`
  - `v_forward = normalize(cross(v_right, v_up))`
- Definir comprimento fixo de pé `L_foot = 0.12 m` na v1.
- `Left_foot = Left_ankle + L_foot * v_forward`
- `Right_foot = Right_ankle + L_foot * v_forward`
- Se `||v_forward|| < 1e-6`, reutilizar `v_forward` válido do frame anterior; no primeiro frame, usar `[0, 0, 1]` no referencial do módulo.

5. Confidence para juntas inferidas
- Spine1/Spine2/Spine3: média das confidences de `pelvis`, `spine`, `thorax`.
- Left/Right_collar: mínimo entre confidence de `Spine3` e ombro correspondente.
- Left/Right_foot: confidence do tornozelo correspondente multiplicada por `0.8`.

6. Juntas redundantes
- Qualquer junta extra da saída original do MotionBERT é descartada no adapter antes do mapper.

#### 5.6.4 Handedness e validações obrigatórias

- Handedness anatômico: left/right sempre do ponto de vista do sujeito, nunca do observador/câmera.
- Validar por frame: `x(Right_hip) - x(Left_hip) > 0` no referencial padronizado; se violar, trocar pares left/right de todos os membros e registrar warning.
- Antes de enviar ao IK, validar:
  - `len(joint_names) == 22`
  - `joint_names` exatamente na ordem da tabela
  - `parents` exatamente igual ao vetor congelado
  - sem NaN em `joint_positions_global_m`

Entrega

Arquivos fixos:

- `configs/skeleton_imugpt22.yaml` (ordem + pais + offsets padrão)
- `configs/motionbert17_contract.yaml` (ordem MB17 esperada no adapter)

### 5.7 metric_normalizer

Responsabilidades
- transformar a saída do MotionBERT (3D normalizado e dependente da câmera) em uma representação consistente para downstream
- separar explicitamente:
- referencial do modelo
- referencial do corpo
- escala métrica aproximada
- preparar dados para o IK sem assumir, prematuramente, um “mundo global real”
#### 5.7.1 Princípio fundamental

A saída do MotionBERT não é métrica nem global. Ela deve ser tratada como:
- pose 3D normalizada
- em um referencial dependente da câmera/imagem
- com escala arbitrária

Portanto, este módulo não “recupera a verdade física”, apenas constrói uma representação consistente e utilizável.

#### 5.7.2 Etapas de transformação
Etapa 1 — Centralização (remoção de translação local instável)

Definir origem no pelvis:

p_i^(centered)(t) = p_i(t) - p_pelvis(t)

Resultado:

- remove translação dependente da câmera
- corpo fica centrado por frame

Etapa 2 — Construção de referencial do corpo (body frame)

Definir eixos por frame:

eixo X (lateral):

x̂ = normalize(Right_hip - Left_hip)

eixo Y (vertical corporal):

ŷ = normalize(Neck - Pelvis)

eixo Z (frente do corpo):

ẑ = normalize(cross(x̂, ŷ))

Re-ortogonalizar se necessário:

ŷ = normalize(cross(ẑ, x̂))

Matriz de rotação:

R_body = [x̂ ŷ ẑ]

Transformação:

p_i^(body)(t) = R_body^T · p_i^(centered)(t)

Resultado:

- remove rotação da câmera
- expressa pose no referencial do corpo

Etapa 3 — Escala métrica aproximada

Impor escala via prior antropométrico:

s = L_real / ||p_joint_a - p_joint_b||

Exemplos:

- fêmur: hip–knee ≈ 0.45 m
- altura total: ≈ 1.70 m

Aplicação:

p_i^(metric_local)(t) = s · p_i^(body)(t)

Importante: isso não recupera escala real, apenas impõe consistência entre clipes

Etapa 4 — Suavização temporal

Aplicar filtro temporal por junta:
- Savitzky–Golay (preferido)
- ou Butterworth low-pass

p_i^(smooth)(t) = smooth(p_i^(metric_local)(t))

Parâmetros típicos:

- cutoff: 5–10 Hz

Objetivo:

- estabilidade para derivadas (IMU)
- remoção de jitter de alta frequência

#### 5.7.3 Saídas do módulo

O módulo deve produzir explicitamente múltiplos níveis:

MetricNormalizationResult = {
    "joint_positions_3d_norm": np.ndarray,         # [T, J, 3] (entrada original)
    "joint_positions_body_frame": np.ndarray,      # [T, J, 3]
    "joint_positions_metric_local": np.ndarray,    # [T, J, 3]
    "joint_positions_smoothed": np.ndarray,        # [T, J, 3]
    "scale_factor": float,
}

Regra:
- nenhuma dessas saídas ainda é “global”

#### 5.7.4 Limitações explícitas
- não há recuperação de profundidade absoluta real
- não há garantia de orientação global correta
- não há trajetória global consistente
- escala é heurística

### 5.8 root_trajectory_estimator

Responsabilidades
- estimar a translação global do corpo (root)
- produzir trajetória consistente para uso no IK/IMUSim
- minimizar drift e inconsistências temporais

#### 5.8.1 Princípio fundamental

O MotionBERT não fornece trajetória global confiável.
Logo, este módulo resolve um problema mal condicionado.

A saída deve ser tratada como aproximação plausível, não ground truth físico.

#### 5.8.2 Estratégia fase 1 (baseline controlado)

Assumir:

- câmera aproximadamente estática
- movimento predominantemente no plano

Definir root como pelvis suavizado:

root(t) = smooth(p_pelvis_original(t)) * s

onde:

- p_pelvis_original vem do espaço normalizado
- s é o fator de escala do metric_normalizer

Opcional:

- planarização (remover drift vertical espúrio)

#### 5.8.3 Estratégia fase 2 (robusta)

Extensões possíveis:

- compensação de movimento de câmera:
  - homografia
  - visual odometry
  - SLAM/SfM leve
- integração com:
  - velocidade estimada por keypoints
  - constraints de contato (foot contact)
- regularização física:
  - suavidade de velocidade
  - limites biomecânicos

#### 5.8.4 Saída
root_translation_m: np.ndarray  # [T, 3]

Propriedades desejadas:
- suavidade temporal
- ausência de saltos
- coerência com pose local

#### 5.8.5 Limitações explícitas
- não é trajetória absoluta real
- sensível a erro de escala
- sensível a oclusão
- drift inevitável sem modelagem de câmera

### 5.9 ik_adapter

Responsabilidades:

- adaptar a nova saída 3D para o mesmo formato usado hoje pela etapa de IK

No artigo, o IK recebe as posições 3D das juntas e a estrutura hierárquica do esqueleto, e produz rotações locais e translação do root.

Interface de entrada
run_ik(
    joint_positions_global_m: np.ndarray,  # [T, J, 3]
    joint_names: List[str],
    parents: List[int],
    fps: int
) -> IKResult
Interface de saída
IKResult = {
    "local_joint_rotations": np.ndarray,   # [T, J, 3 or 4]
    "root_translation_m": np.ndarray,      # [T, 3]
    "bvh_path": Optional[str],
}
Regra

O restante do código IMUGPT não deve perceber se a pose veio do T2M-GPT ou do vídeo.

### 5.10 imusim_adapter

Responsabilidades:

- reusar o contrato atual do IMUSim
- associar sensores a segmentos corporais

No artigo, o IMUSim calcula aceleração e velocidade angular a partir das rotações locais e da translação do root, com sensores virtuais em 22 localizações on-body e com adição de ruído.

Entrada
- local_joint_rotations
- root_translation_m
- sensor_layout.yaml
Saída
VirtualIMUSequence = {
    "acc": np.ndarray,      # [T, S, 3]
    "gyro": np.ndarray,     # [T, S, 3]
    "sensor_names": List[str],
    "fps": int,
}
## 6. Contratos de dados

### 6.1 Convenções geométricas

#### 6.1.1 Separação de referenciais (obrigatório)

O pipeline deve distinguir explicitamente:

1. Model frame (MotionBERT output)
  - coordenadas normalizadas
  - dependentes da câmera
  - escala arbitrária
2. Body frame
  - origem no pelvis
  - orientação definida pelo corpo
  - independente da câmera
3. Metric local frame
  - body frame + escala antropométrica
  - unidade: metros (aproximados)
4. Pseudo-global frame
  - metric local + root_translation
  - usado pelo IK e IMUSim

#### 6.1.2 Convenção final (para IK / IMUSim)

Sistema destro:
- x: lateral (positivo → direita do sujeito)
- y: vertical (positivo → cima)
- z: anterior (positivo → frente do sujeito)

Unidade: metros

#### 6.1.3 Representação final obrigatória

Antes do IK, os dados devem satisfazer:

joint_positions_global_m: np.ndarray  # [T, J, 3]
root_translation_m: np.ndarray        # [T, 3]

onde:

joint_positions_global_m(t) = root_translation_m(t) + joint_positions_metric_local(t)

#### 6.1.4 Regras obrigatórias
- sistema destro consistente em todo o pipeline
- left/right anatômico (sujeito)
- nenhuma junta com NaN
- ordem de joints fixa (IMUGPT22)
- pais exatamente iguais ao contrato

#### 6.1.5 Observação crítica

Mesmo após todas as etapas:

- o sistema resultante é pseudo-métrico
- não representa reconstrução física absoluta
- é adequado para:
  - IK
  - síntese de IMU
  - HAR
- mas não para:
  - reconstrução métrica precisa
  - análise biomecânica clínica

### 6.2 Convenções temporais
- fps_pose = fps_imu = 20 na fase de exportação para IMUSim
- timestamps preservados para debug
- reamostragem somente após pose 3D limpa
### 6.3 Convenções de confiança

Toda junta deve carregar confidence 0..1.

Uso da confidence:

- mascarar interpolação
- ponderar suavização
- pontuar qualidade do clipe
## 7. Estrutura de código recomendada
imugpt/
  pose_module/
    __init__.py
    pipeline.py
    interfaces.py
    io/
      video_loader.py
      cache.py
    tracking/
      person_selector.py
    vitpose/
      estimator.py
      adapter.py
    motionbert/
      lifter.py
      adapter.py
    processing/
      cleaner2d.py
      temporal_filters.py
      skeleton_mapper.py
      metric_normalizer.py
      root_estimator.py
      quality.py
    export/
      ik_adapter.py
      debug_video.py
    configs/
      pose_module.yaml
      skeleton_imugpt22.yaml
      sensor_layout.yaml
## 8. API pública recomendada
def generate_pose_from_video(
    video_path: str,
    clip_id: str,
    fps_target: int = 20,
    save_debug: bool = True,
    config_path: str = "pose_module/configs/pose_module.yaml"
) -> PoseSequence3D:
    ...
API de ponta a ponta para o IMUGPT
def generate_virtual_imu_from_video(
    video_path: str,
    clip_id: str,
    fps_target: int = 20
) -> VirtualIMUSequence:
    pose3d = generate_pose_from_video(video_path, clip_id, fps_target)
    ik = run_ik(...)
    imu = run_imusim(...)
    return imu
## 9. Configuração mínima
pose_module:
  mode: offline
  fps_target: 20
  single_person_only: true
  save_intermediate: true

vitpose:
  model_size: base
  device: cuda
  batch_size: 16
  keypoint_format: coco

motionbert:
  checkpoint: pretrained
  window_size: 81
  window_overlap: 0.5
  device: cuda

cleaning:
  max_gap_interp: 5
  savgol_window: 9
  savgol_polyorder: 2
  low_conf_threshold: 0.2

normalization:
  target_height_m: 1.70
  align_body_axes: true
  smooth_root: true
  # Parâmetros adicionais para suavidade temporal
  temporal_filter: savgol
  filter_cutoff_hz: 7.0

quality:
  min_visible_joint_ratio: 0.8
  max_outlier_ratio: 0.1
## 10. Política de qualidade

Cada clipe deve gerar um relatório:

{
  "clip_id": "...",
  "status": "ok|warning|fail",
  "visible_joint_ratio": 0.93,
  "mean_confidence": 0.81,
  "temporal_jitter_score": 0.12,
  "root_drift_score": 0.08,
  "skeleton_mapping_ok": true,
  "notes": []
}
Regras de aceite

Aceitar o clipe para IMUSim apenas se:

- visible_joint_ratio >= 0.80
- mean_confidence >= 0.50
- temporal_jitter_score <= threshold
- mapping_ok == true
## 11. Plano de validação

### 11.1 Validação técnica do módulo de pose

Medir:

- estabilidade temporal das juntas
- continuidade do root
- taxa de frames válidos
- taxa de IK sem falha
- plausibilidade biomecânica
Testes mínimos
- vídeo frontal de caminhada
- vídeo lateral de corrida
- subir/descer escadas
- sentar/levantar
- clipe com oclusão parcial
- clipe com câmera móvel leve
### 11.2 Validação da síntese de IMU

Comparar IMU sintetizada contra IMU real quando houver sincronização:

- correlação por eixo
- erro espectral
- DTW
- consistência de picos de aceleração
### 11.3 Validação final no HAR

Comparar três condições:

- real only
- real + virtual(T2M-GPT original)
- real + virtual(video pose module)

O artigo mostra que misturar dados reais com IMU virtual melhora o desempenho sobre treinar apenas com IMU real em vários cenários. Esse deve ser o critério final de sucesso do novo módulo.

## 12. Riscos principais
Risco 1 — 2D muito bom, 3D ruim

ViTPose melhora a pose 2D, mas a qualidade final depende fortemente do lifting 3D. Isso é uma inferência de engenharia; o MotionBERT resolve 3D pose estimation a partir de skeletons 2D, mas não elimina problemas de câmera, escala e oclusão.

Risco 2 — esqueleto incompatível

O T2M-GPT no pipeline original produz 22 juntas, então o novo mapeamento precisa reproduzir esse contrato com rigor.

Risco 3 — root drift

Sem compensação de câmera, a aceleração sintetizada pode ficar fisicamente incorreta.

Risco 4 — ruído explode nas derivadas

Sem suavização temporal adequada, o IMUSim receberá cinemática instável.

## 13. Fases de implementação
Fase 1 — Integração mínima viável
- ViTPose
- limpeza 2D
- MotionBERT
- mapper para esqueleto IMUGPT22
- normalização métrica simples
- export para IK
- debug video
Fase 2 — Robustez
- tracking melhor
- compensação de câmera
- confidence-aware smoothing
- melhor estimação de escala
Fase 3 — Avaliação HAR
- gerar IMU virtual em lote
- calibrar
- treinar RF / DeepConvLSTM
- comparar com baseline original
## 14. Decisões recomendadas
- Não alterar o IK nem o IMUSim na primeira versão.
- Criar um adapter estrito para que a saída do módulo de pose imite a saída esperada do T2M-GPT.
- Fixar o esqueleto-alvo em 22 juntas desde o início.
- Adotar single-person only como restrição oficial da v1.
- Tratar a reconstrução de root global como componente crítico, não acessório.
## 15. Especificação final resumida

O módulo de pose a integrar ao IMUGPT deve:

- receber vídeo monocular
- estimar pose 2D com ViTPose
- limpar e padronizar a sequência 2D
- levantar para 3D com MotionBERT em janelas temporais
- mapear para o esqueleto IMUGPT-22
- converter para coordenadas globais métricas consistentes
- estimar root_translation_m
- exportar para o IK exatamente no formato esperado pelo pipeline atual
- deixar IK → IMUSim → calibração → HAR inalterados, como no fluxo descrito no artigo/repositório original.
[file content end]