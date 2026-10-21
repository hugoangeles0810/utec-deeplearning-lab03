# UTEC Deep Learning · Lab 03: AHCL para re-identificación de personas

Implementación propia de **AHCL** (*Attention-based hybrid contrastive learning for
unsupervised person re-identification*, Qin et al., Scientific Reports 15:13238,
2025; [`docs/paper.pdf`](docs/paper.pdf)) adaptada al dataset del laboratorio, con
evaluación sobre queries originales y queries de baja resolución.

> **Estado:** el método, el modelo base, la ablación, la evaluación y el notebook
> de análisis están implementados y probados. Las tablas de resultados se
> completan con la salida de [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)
> (sección 5) una vez entrenados los runs.

## 1. El paper

### Propuesta central

La re-identificación no supervisada entrena con pseudo-etiquetas obtenidas al
agrupar los embeddings, así que su calidad depende de la calidad de las
características. AHCL ataca el problema desde dos frentes:

1. **Mejores características**, mediante el módulo de atención **CAPF** al final de
   ResNet-50.
2. **Una señal contrastiva más rica**, mediante una pérdida **híbrida** que compara
   cada muestra contra los centroides de los clusters (estable) y contra
   instancias difíciles (discriminativa).

### Componentes

| Componente | Paper | Código |
|---|---|---|
| Encoder | ResNet-50 (ImageNet) sin capas tras `layer4`; *last stride* 1 | `model.build_backbone` |
| Atención espacial | avg y max sobre canales → conv 7×7 → sigmoide → máscara (ec. 2) | `model.SpatialAttention` |
| Fusión de pooling | GAP y GMP, cada uno con su bloque SE, sumados | `model.PoolingFusion` |
| Reducción de dimensión | FC → BN → L2, 2048-d | `model.EmbeddingHead` |
| Pseudo-etiquetas | distancia de Jaccard k-recíproca (k1 = 30) + DBSCAN (eps = 0.6, min = 4) | `clustering.py` |
| Memoria de clusters `Mc` | InfoNCE contra centroides (ec. 4), actualización con momento (ec. 5) | `memory.ClusterMemory` |
| Memoria de instancias `Mi` | positivo = la instancia más lejana del propio cluster, negativos = la más cercana de cada otro cluster del batch (ec. 6), reemplazo (ec. 7) | `memory.InstanceMemory` |
| Pérdida híbrida | `L = (1 − a)·L_ccl + a·L_icl`, a = 0.5 (ec. 3) | `memory.HybridMemory` |
| Muestreo | P = 16 pseudo-identidades × Z = 16 imágenes | `sampler.PseudoLabelBatchSampler` |
| Entrenamiento | por época: extraer → agrupar → reconstruir memorias → entrenar (Alg. 1). Adam, lr 3.5e-4, wd 5e-4, ×0.1 cada 20 épocas, 50 épocas | `train.py` |

Todo el método está implementado en este repositorio, con PyTorch, torchvision
(ResNet-50 y el bloque SE) y scikit-learn (DBSCAN). No se usa ninguna
implementación existente de AHCL ni de SPCL.

## 2. Datos

| Split | Contenido | Uso |
|---|---|---|
| `training.h5` | 11 043 imágenes **sin identidad**, 6 cámaras | entrenamiento no supervisado |
| `validation.h5` | 100 queries (100 ids) + 1 793 imágenes de galería | selección de configuración y de época |
| `testing.h5` | 3 368 queries (750 ids) + 19 732 imágenes de galería | evaluación final, solo una vez por run |

Todas las imágenes son RGB de 128×64. Cada query tiene además una versión de
**32×16** (`images_low_resolution`). En la galería de test, el id `-1` marca
imágenes basura, que se ignoran, y el id `0` marca distractores, que cuentan
como negativos. Es la estructura de Market-1501.

## 3. Adaptaciones y decisiones

| Aspecto | Paper | Aquí | Justificación |
|---|---|---|---|
| Resolución de entrada | 256×128 | **128×64** | Es la resolución nativa del dataset; escalar al doble no agrega información. Con *last stride* 1 el mapa final es 8×4. |
| Queries de baja resolución | no se tratan | 32×16 → **128×64 bicúbica** antes de la red (`transforms.build_low_res_transform`) | Permite evaluar las dos condiciones con el mismo modelo y la misma galería, sin reentrenar. |
| Métrica de recuperación | distancia euclídea | **similitud coseno** entre embeddings L2 | La exige la tarea. Para vectores unitarios, ‖a − b‖² = 2 − 2·cos(a, b), así que el ranking es idéntico. |
| Actualización de `Mc` | ec. 5; el texto no dice si es por batch o por época | momento m = 0.2 **por batch**, y la memoria se reconstruye tras cada clustering | Es la lectura estándar en SPCL y Cluster-Contrast, en los que el paper dice basarse. |
| Negativos de `L_icl` | "P identidades" del batch | la instancia más cercana de cada **cluster presente en el batch** | Es la ec. 6 literal con P = 16 clusters por batch. |
| Selección de modelo | no se detalla | `best.pt` = época con mayor mAP de validación (queries originales), evaluada cada 5 épocas | La tarea pide seleccionar con validación; el test nunca se mira para decidir. |
| Modelo base | "ResNet-50 con memoria de instancias" | `--no-capf --loss-weight 1`: ResNet-50 + GAP + BN + solo `L_icl` | Es la variante más cercana que permite el código. La memoria de instancias del paper no está especificada en detalle, así que usamos nuestra `L_icl`. |

Augmentaciones de entrenamiento, como en SPCL: volteo horizontal, *pad* 10 +
recorte aleatorio y *random erasing* (p = 0.5).

## 4. Protocolo experimental

- **Entrenamiento:** solo con `training.h5`, sin etiquetas.
- **Selección:** las variantes y los hiperparámetros (`--eps`, `--loss-weight`,
  `--k1`, `--lr`…) se comparan por el mejor mAP de validación de cada run
  (`history.json`).
- **Modelo base:** `runs/baseline`. **Ablación:** se quita un componente cada vez,
  como en las tablas 4 y 5 del paper.

  | Run | Flags | Qué mide |
  |---|---|---|
  | `ahcl` | ninguno | el método completo |
  | `baseline` | `--no-capf --loss-weight 1` | el modelo base |
  | `abl_no_sa` | `--no-spatial-attention` | el aporte de la atención espacial |
  | `abl_no_se` | `--no-se` | el aporte de la atención de canal (SE) |
  | `abl_only_ccl` | `--loss-weight 0` | `L_ccl` sola (sin `L_icl`) |

- **Test:** el `best.pt` de cada run se evalúa una sola vez en dos condiciones.
  Ambas usan el **mismo modelo**, embeddings normalizados L2 y similitud coseno:
  1. **Original:** queries y galería en 128×64.
  2. **Baja resolución:** queries en 32×16 y galería original.
- **Métricas:** mAP, Rank-1, Rank-5 y Rank-10 (CMC) con el protocolo de
  Market-1501, que excluye de la galería las imágenes de la misma identidad y
  cámara que la query. Se reporta también la **caída de mAP** absoluta,
  `mAP_orig − mAP_low`, y relativa, `(mAP_orig − mAP_low) / mAP_orig`.

## 5. Cómo reproducir

```sh
uv sync
# Datos: ver data/README.md (training.h5, validation.h5, testing.h5 en data/)

uv run lab03 train --out runs/ahcl
uv run lab03 train --no-capf --loss-weight 1 --out runs/baseline
uv run lab03 train --no-spatial-attention --out runs/abl_no_sa
uv run lab03 train --no-se --out runs/abl_no_se
uv run lab03 train --loss-weight 0 --out runs/abl_only_ccl

for run in ahcl baseline abl_no_sa abl_no_se abl_only_ccl; do
  uv run lab03 evaluate runs/$run/best.pt   # escribe runs/$run/best_testing.json
done

uv run jupyter lab notebooks/analysis.ipynb
```

`lab03 train` escribe en `--out` los archivos `config.json`, `history.json`
(métricas por época), `last.pt` y `best.pt`. Con `uv run lab03 train --help` se
listan todos los flags.

## 6. Resultados

> Pendiente: pegar las tablas de la sección 5 del notebook.

### Validación (mejor época de cada run)

| Run | Variante | Época | mAP | R1 | mAP baja res. | Caída |
|---|---|---|---|---|---|---|
| ahcl | AHCL (paper) | — | — | — | — | — |
| baseline | sin CAPF, a=1 | — | — | — | — | — |

### Test

| Run | Variante | mAP | R1 | R5 | R10 | mAP baja res. | R1 | R5 | R10 | Caída de mAP |
|---|---|---|---|---|---|---|---|---|---|---|
| ahcl | AHCL (paper) | — | — | — | — | — | — | — | — | — |
| baseline | sin CAPF, a=1 | — | — | — | — | — | — | — | — | — |
| abl_no_sa | sin atención espacial | — | — | — | — | — | — | — | — | — |
| abl_no_se | sin SE | — | — | — | — | — | — | — | — | — |
| abl_only_ccl | a=0 | — | — | — | — | — | — | — | — | — |

## 7. Análisis de errores

El notebook ([`notebooks/analysis.ipynb`](notebooks/analysis.ipynb), sección 4)
analiza el modelo elegido en test con lo siguiente:

- **Distribución del AP por query** en ambas condiciones, y la dispersión del AP
  original frente al de baja resolución. Muestra si la caída se concentra en pocas
  queries o afecta a todas.
- **Rank del primer acierto:** rank 1, 2–5, 6–10 o más de 10.
- **Sesgo de cámara:** en los errores de Rank-1, qué fracción del primer resultado
  viene de la misma cámara que la query, comparada con la proporción esperada al
  azar.
- **Consistencia entre resoluciones:** similitud coseno entre el embedding de cada
  query y el de su versión 32×16, y su correlación con la pérdida de AP.
- **Ejemplos visuales:** top-10 de las queries que más AP pierden con baja
  resolución y de las peores en resolución original.

> Pendiente: resumir aquí los hallazgos con los números del notebook.

## 8. Limitaciones

- **Ruido de las pseudo-etiquetas:** todo depende de DBSCAN (`eps`, `k1`). Los
  outliers no se entrenan en esa época, y los errores de clustering iniciales se
  refuerzan.
- **Sesgo de cámara:** el método no tiene ningún componente que tenga en cuenta la
  cámara. Las imágenes de una misma cámara comparten fondo e iluminación y tienden
  a agruparse juntas.
- **Baja resolución:** AHCL no tiene ningún mecanismo para la resolución. La red
  nunca ve imágenes de 32×16 durante el entrenamiento, y el reescalado bicúbico no
  recupera los detalles perdidos. Una extensión natural, que la tarea permite, es
  aumentar el entrenamiento con versiones reducidas y vueltas a ampliar. No se
  aplicó, para no alterar el método evaluado.
- **Atención espacial a 128×64:** el mapa final es de 8×4 y la convolución de la
  atención espacial es de 7×7, así que cada peso de la máscara ve casi todo el
  mapa. La atención es mucho menos local que a 256×128, como en el paper, y su
  aporte puede ser menor.
- **Validación pequeña:** con 100 queries, cada query vale 1 punto de Rank-1. Las
  diferencias pequeñas entre configuraciones no son significativas, y cada run usa
  una sola semilla.
- **Costo:** la distancia de Jaccard mantiene matrices densas de N×N. Con
  N = 11 043 son unos 0.5 GB cada una, y crece de forma cuadrática con el
  dataset.

## Estructura

```
src/lab03/
  model.py        ResNet-50 + CAPF (y variantes para modelo base y ablación)
  clustering.py   distancia de Jaccard k-recíproca + DBSCAN
  memory.py       memorias de clusters e instancias, pérdida híbrida
  sampler.py      batches P×Z por pseudo-etiqueta
  train.py        bucle de entrenamiento (Alg. 1)
  evaluate.py     embeddings, distancia coseno, CMC/mAP, AP por query, caída de mAP
  analysis.py     lectura y tablas de los JSON de los runs
  main.py         CLI `lab03 train` / `lab03 evaluate`
notebooks/
  dataset.ipynb   exploración del dataset
  analysis.ipynb  comparación de runs y análisis de errores
tests/            pytest (uv run pytest)
```

Desarrollo: `uv run pytest`, `uv run ruff check .` y `uv run ruff format .`.
