# OpenProof

Minimal mainline training and evaluation for OpenProof. The implementation has
one model: frozen RF-DETR Small, trajectory objectness, QAM, full IRM (identity
routing and learned redundancy weighting), and full REM. Training and inference
use the same spatially masked Odds rule. There are no ablation switches.

## Installation

Use Linux or WSL Ubuntu, Python 3.10, and a CUDA-enabled PyTorch installation.
The local validation environment uses PyTorch 2.11.0 / torchvision 0.26.0 with
CUDA 12.8, RF-DETR 1.9.3, NumPy 2.2.6, SciPy 1.15.3, and pycocotools 2.0.8.

```bash
python -m pip install -e .
python scripts/download_pretrained.py
```

The initialization is the official RF-DETR Small detector checkpoint. It
contains detection pretraining, rather than an ImageNet-only backbone. Each
first task initializes fresh Known classifier rows while retaining the detector's
backbone, box decoder and box heads.

## Fixed training recipe

| Item | Configuration |
|---|---|
| RF training | New classes: 4 consecutive epochs |
| Incremental RF replay | Task2–4: the official permitted replay pool, 1 subsequent epoch |
| RF batch / accumulation | 8 / 2; effective batch 16 |
| RF optimizer | Official RF-DETR training; lr 1e-4, encoder lr 1.5e-4; no EMA or early stopping |
| OPEN budget | 1035 AdamW updates; microbatch 16; cover the complete permitted pool |
| OPEN learning rates | Objectness / QAM: 1e-4; relation head: 1e-3 |
| Weight decay / clipping | 1e-4 / 20 |
| QAM hidden layers | Frozen after update312; readouts remain trainable |
| Identity verification | Full exemplars, two image folds, distance quantiles, absolute distance + competing-class margin; three Known-fitted scalars |
| Features / candidates | 512 input; 300 decoder queries; FP32 frozen features |
| IRM | Routing enabled; learned redundancy weighting; temperature1 |
| Ranking | Joint candidate×class TopK; no per-query argmax, score gate or post-hoc NMS |
| Seed | 14022 |

Seven OPEN objectives retain their original targets and reductions: objectness
BCE, objectness ranking, localization BCE, completeness BCE, same-instance BCE,
joint quality ranking, and continuous related-quality ranking. The last has
weight0.5; the other six have weight1.

Across incremental tasks the classifier expands semantically, including moving
the background row. Old-class logits are masked only during new-class RF
training. Shared OPEN parameters inherit from the preceding task. Current RF
features and the complete current Known reference are rebuilt; old feature
spaces and reference tensors are not mixed. A frozen preceding model protects
physical targets outside current annotated support.

## Data layout

Dataset images and annotations are not included. Obtain each benchmark's
official splits and replay lists. `configs/` specifies exact category order,
test sizes and annotation filenames. Set `data_root`, `pretrained`, `output`,
and image paths for your installation.

```text
data/
  images/                         # relative file_name paths resolve here
  vehicle/split1/{train,test}.json # repeat split2–4; Animal uses the same layout
  animal/split1/{train,test}.json
  voccoco/{train,voc,coco_mixed,coco_ood}.json
  M-OWODB/task1/{new,open,test}.json
  M-OWODB/task2/{new,replay,open,test}.json # repeat task3–4
  M-OWODB/Annotations/*.xml
  S-OWODB/...                      # analogous task1–4 layout
weights/rf-detr-small.pth
```

Annotation files use standard COCO `images`, `annotations`, and `categories`.
Category names are mapped to the configuration's ordered Known classes;
non-Known labels become Unknown only in test annotations. VOC/COCO name aliases
are normalized. Image identifiers must be unique, including across VOC and COCO
namespaces. Keep full test images, including negative images.

For OWOD, `new.json` is the official new-class list, `replay.json` is the
official permitted replay pool, and `open.json` is the permitted annotated OPEN
training pool for that task. Do not substitute a union of all old full training
sets. Test images retain `owod_id`, matching the original full XML filename.
Original XML files preserve difficult flags and future-category annotations for
the author's test evaluation; they are not inputs to training.

`scripts/convert_voc.py` converts an official image list and XML annotations to
this COCO structure while keeping differently padded identifiers distinct:

```bash
python scripts/convert_voc.py --image-list task1_train.txt \
  --xml-root /datasets/OWOD/Annotations --images-root /datasets/OWOD/JPEGImages \
  --output data/M-OWODB/task1/new.json
```

The converter emits stable namespace-safe IDs; use one consistent annotation
export for all runs. Training monitors are fixed slices of training data,
not independent validation sets. Final RF checkpoints use the fixed budget,
not test or monitor selection.

## Unified commands

For any benchmark substitute its config: `vehicle.yaml`, `animal.yaml`,
`voccoco.yaml`, `m-owodb.yaml`, or `s-owodb.yaml`.

```bash
# Prepare annotations and validate the supplied protocol.
python -m openproof prepare --config configs/m-owodb.yaml

# Train the full chain; completed stages are reused, partial stages resume.
python -m openproof train --config configs/m-owodb.yaml

# Evaluate the trained chain, without training.
python -m openproof test --config configs/m-owodb.yaml

# Or run preparation, training and testing in order with one command.
python -m openproof run --config configs/m-owodb.yaml

# One independent OpenImages split:
python -m openproof run --config configs/vehicle.yaml --task split1

# Only one already-trained task's test:
python -m openproof test --config configs/m-owodb.yaml --task task4
```

Vehicle/Animal splits are independent. VOC, COCO-Mixed and COCO-OOD share one
trained detector/OPEN/reference. M and S each maintain a separate inherited
four-task chain. Class-disjoint QAM readout episodes are used for VOC/COCO and
OWOD, not for OpenImages. Geometry-aware encoder selection is used only for
frozen OWOD feature extraction and inference; RF training uses native selection.

Training writes native RF recovery checkpoints and OPEN optimizer/RNG/cursor
state. Completed OPEN training removes its rebuildable feature cache and
duplicate last checkpoint, preserving the final RF, OPEN, reference and tests.

## Evaluation and outputs

AP follows each benchmark protocol. Scores are fractions; multiply by100 for tables.

- OpenImages: jointTop100; standard COCO **101-point** evaluation. Known/Unknown
  AP averages IoU .50:.95; AP50 uses IoU .50. Crowd/ignore and maxDets100 per
  image/category are retained.
- VOC/COCO: jointTop500. Known AP and AP50 use standard **COCO101**. Unknown
  AP50 uses the author **VOC07 11-point** evaluator on images containing Unknown
  GT. Mixed retains897 total images and its original704-image Unknown subset.
  VOC Unknown metrics are N/A.
- M/S-OWODB: jointTop500; **VOC2010 continuous/all-points AP** at IoU >.50, with
  original inclusive-pixel matching, difficult handling, 3-decimal score and
  1-decimal coordinate serialization. Old/new/all Known AP50, Unknown AP50,
  U-Recall, H, A-OSE and WI are saved. Task4 Unknown/H/WI are N/A.

Metric fields use `K_AP`/`K_AP50` and `U_AP`/`U_AP50`. Each `metrics.json` includes
`AP_protocol` with the actual Known and Unknown interpolation, IoU and matching
rules. Unknown IoU-averaged `U_AP` is N/A for VOC/COCO.

Saved complete predictions can be rescored on CPU without training or inference:

```bash
python -m openproof score --config configs/m-owodb.yaml --device cpu
```

Substitute the desired benchmark config. Old metrics with a different protocol
are retained as `metrics_before_protocol_fix.json` before corrected metrics are
written to `metrics.json`.

```text
runs/<benchmark>/<split-or-task>/
  rf.pth, rf_completed.json, classes.json
  reference.pth, open_final.pth, open_completed.json, training_config.json
  tests/<test>/{manifest.json,shards/,metrics.json}
runs/<benchmark>/results.json
```

## Checks

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

The extraction was compared numerically with the retained implementation for
REM, routing/Odds, all seven losses and their gradients. Real-image frozen RF
feature capture was also checked on CPU. These are implementation checks, not a
new accuracy run. The RF4 + Routing-on default is the requested release recipe;
historical tables with other detector budgets/configurations are not bundled as
new results for this recipe.

See `THIRD_PARTY_NOTICES.md` for evaluator provenance. No datasets, trained
weights, caches, experiment queues, ablation scripts or historical result files
are part of this archive.
