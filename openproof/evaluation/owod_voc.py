# Copyright (c) Facebook, Inc. and its affiliates.
# Pascal VOC evaluator lineage from Detectron2 (Apache License 2.0).
"""Pascal evaluation functions from OrthogonalDet / Detectron2 OWOD evaluator.
Preserved matching, difficult handling, score serialization and open-set counts.
Only the removed NumPy bool alias is updated.
"""
import logging
import numpy as np
import xml.etree.ElementTree as ET
from functools import lru_cache
class PathManager:
    @staticmethod
    def open(path,mode='r'):return open(path,mode)
@lru_cache(maxsize=None)
def parse_rec(filename, known_classes):
    """Parse a PASCAL VOC xml file."""
    VOC_CLASS_NAMES_COCOFIED = ['airplane', 'dining table', 'motorcycle', 'potted plant', 'couch', 'tv']
    BASE_VOC_CLASS_NAMES = ['aeroplane', 'diningtable', 'motorbike', 'pottedplant', 'sofa', 'tvmonitor']
    try:
        with PathManager.open(filename) as f:
            tree = ET.parse(f)
    except:
        logger = logging.getLogger(__name__)
        logger.info('Not able to load: ' + filename + '. Continuing without aborting...')
        return None
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        cls_name = obj.find('name').text
        if cls_name in VOC_CLASS_NAMES_COCOFIED:
            cls_name = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls_name)]
        if cls_name not in known_classes:
            cls_name = 'unknown'
        obj_struct['name'] = cls_name
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, known_classes=None):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    with PathManager.open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    imagenames_filtered = []
    recs = {}
    mapping = {}
    for imagename in imagenames:
        rec = parse_rec(annopath.format(imagename), tuple(known_classes))
        if rec is not None and int(imagename) not in mapping:
            recs[imagename] = rec
            imagenames_filtered.append(imagename)
            mapping[int(imagename)] = imagename
    imagenames = imagenames_filtered
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[mapping[int(image_ids[d])]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih
            uni = (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.0
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    "\n    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)\n                                    ===========    \n    Absolute OSE = # of unknown objects classified as known objects of class 'classname'\n    WI = FP_openset / (TP_closed_set + FP_closed_set)\n    "
    logger = logging.getLogger(__name__)
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == 'unknown']
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}
    if classname == 'unknown':
        return (rec, prec, ap, 0, n_unk, None, None)
    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[mapping[int(image_ids[d])]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih
            uni = (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            is_unk[d] = 1.0
    is_unk_sum = np.sum(is_unk)
    tp_plus_fp_closed_set = tp + fp
    fp_open_set = np.cumsum(is_unk)
    return (rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set)
