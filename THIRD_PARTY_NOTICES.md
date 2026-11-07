# Third-party components

- RF-DETR (`rfdetr==1.9.3`) is installed as a dependency, not vendored.
- `openproof/evaluation/unknown_voc.py` is UnSniffer's in-memory evaluator;
  its source header credits Muhammet Bastan and Bharath Hariharan / Fast R-CNN.
  The upstream repository license is retained under `licenses/`.
- `openproof/evaluation/owod_voc.py` contains only the Pascal matching/AP
  functions from OrthogonalDet's `core/pascal_voc_evaluation.py` (Detectron2 / OWOD
  evaluator lineage), with a local file opener and the NumPy bool alias updated.
  Detectron2's Apache 2.0 license is retained in `licenses/Detectron2-APACHE-2.0.txt`.
- COCO matching uses pycocotools. Eleven-point recall sampling is set explicitly.

Dataset images, annotations and pretrained weights are obtained separately from
their respective providers and are not redistributed in this code archive.
