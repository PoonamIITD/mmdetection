"""
Thin wrapper around the REAL official CityPersons eval classes
(coco.py / eval_MR_multisetup.py, copied verbatim from
cvgroup-njust/CityPersons, patched only for Python-3/modern-numpy syntax --
no algorithmic changes).

What this module does that the official files don't:
  1. Converts your single multi-category COCO GT json (with
     ann["attributes"]["visibility_ratio"]) into the flat single-category
     format the official code requires: every GT box gets a top-level
     'height', 'vis_ratio', and pre-set 'ignore' field, and all boxes
     (target-category or not) are remapped to a single category_id=1 so
     they participate in the same matching/ignore pool.
  2. Same remap for prediction jsons (drops non-target-category dets).
  3. A tiny COCOeval subclass that additionally stores the FULL fppi/miss-
     rate curve (the stock accumulate() only keeps the 9 sampled FPPI
     points needed for the MR^-2 scalar, not the full curve needed for a
     log-log plot).
  4. Support for extra custom subsets by appending to params.HtRng/VisRng/
     SetupLbl instead of only the 4 official id_setup indices 0-3.
"""

import sys
import json
import copy
from pathlib import Path
import numpy as np

# from pycocotools.coco import COCO                      
from eval_MR_multisetup import COCOeval as _OfficialCOCOeval


# ── GT / prediction conversion ───────────────────────────────────────────────

def convert_gt_to_official_format(gt_full, target_category_ids, out_path):
    """
    Official _prepare() expects, per annotation:
        ann['height']    -> bbox height (float)
        ann['vis_ratio'] -> visibility ratio (float)
        ann['ignore']    -> 0/1, PRE-SET before the height/vis window check
                             (_prepare() only ever promotes 0->1, never
                             demotes 1->0, when the box is inside the
                             current subset's window)
        ann['category_id'] == 1  (the single merged "person" class --
                             accumulate() hardcodes catIds=[1])
    """
    new_anns = []
    next_id = 1
    for ann in gt_full["annotations"]:
        is_target = ann["category_id"] in target_category_ids
        vis = ann.get("attributes", {}).get("visibility_ratio")
        h = ann["bbox"][3]

        if (not is_target) or (vis is None):
            ignore = 1
            vis_ratio = vis if vis is not None else 1.0  # value irrelevant once ignore=1
        else:
            ignore = 0
            vis_ratio = vis

        new_anns.append({
            "id": next_id,
            "image_id": ann["image_id"],
            "category_id": 1,
            "bbox": ann["bbox"],
            "height": h,
            "vis_ratio": vis_ratio,
            "ignore": ignore,
        })
        next_id += 1

        converted = {
            "info": {},
            "licenses": [],
            "images": gt_full["images"],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": new_anns,}
    with open(out_path, "w") as f:
        json.dump(converted, f)
    return out_path


def convert_preds_to_official_format(pred_list, target_category_ids):
    """
    category_id -> 1 for target-category detections, everything else dropped.
 
    IMPORTANT: 'height' is set explicitly here rather than relying on
    loadRes() to add it. The CityPersons-forked coco.py's loadRes() does
    `ann['height'] = bb[3]` for bbox results -- that line is the ONLY
    reason that fork exists; stock pycocotools.coco.COCO.loadRes() does
    NOT set it, and evaluateImg() reads d['height'] for the expFilter DT
    height cull, so omitting it causes a KeyError under plain pycocotools.
    loadRes() never overwrites a field that's already present, so setting
    it here works identically under both coco.py and pycocotools.
    """
    out = []
    for p in pred_list:
        if p["category_id"] not in target_category_ids:
            continue
        out.append({
            "image_id": p["image_id"],
            "category_id": 1,
            "bbox": p["bbox"],
            "score": p["score"],
            "height": p["bbox"][3],
        })
    return out


# ── COCOeval subclass that also keeps the full curve ─────────────────────────

class COCOevalWithCurve(_OfficialCOCOeval):
    """
    Identical to the official accumulate(), specialized for our case
    (T=len(iouThrs)==1, K=1 category, M=1 maxDets) so the loop collapses to
    a single (t, k, m) triple, plus it additionally stores the full
    (fppi, miss_rate) arrays used for plotting, in self.full_curve.
    Nothing about the matching/ranking logic is changed -- this only adds
    bookkeeping around the official cumsum step.
    """

    def accumulate(self, p=None):
        if not self.evalImgs:
            raise RuntimeError("Please run evaluate() first")
        if p is None:
            p = self.params
        p.catIds = [1]

        _pe = self._paramsEval
        setI = set(_pe.imgIds)
        i_list = [n for n, i in enumerate(_pe.imgIds) if i in setI]
        I0 = len(_pe.imgIds)

        E = [self.evalImgs[i] for i in i_list]
        E = [e for e in E if e is not None]

        T = len(p.iouThrs)
        R = len(p.fppiThrs)
        ys = -np.ones((T, R, 1, 1))
        self.full_curve = None

        if len(E) > 0:
            dtScores = np.concatenate([e["dtScores"][0:p.maxDets[-1]] for e in E])
            inds = np.argsort(-dtScores, kind="mergesort")
            dtm = np.concatenate([e["dtMatches"][:, 0:p.maxDets[-1]] for e in E], axis=1)[:, inds]
            dtIg = np.concatenate([e["dtIgnore"][:, 0:p.maxDets[-1]] for e in E], axis=1)[:, inds]
            gtIg = np.concatenate([e["gtIgnore"] for e in E])
            npig = np.count_nonzero(gtIg == 0)

            if npig > 0:
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                keep = np.where(dtIg == 0)[1]
                tps = tps[:, keep]
                fps = fps[:, keep]

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    recall = (tp / npig).tolist()
                    fppi = (fp / I0).tolist()

                    # official monotonic-envelope smoothing step (verbatim)
                    for i in range(len(recall) - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]

                    q = np.zeros((R,)).tolist()
                    fppi_arr = np.array(fppi)
                    ridx = np.searchsorted(fppi_arr, p.fppiThrs, side="right") - 1
                    for ri, pi in enumerate(ridx):
                        if pi >= 0:
                            q[ri] = recall[pi]
                        # else: leave at 0 (miss-rate=1) -- see note in the
                        # ported-script version about the official code's
                        # negative-index wraparound quirk here.
                    ys[t, :, 0, 0] = np.array(q)

                    # store the FULL curve (leading virtual point included)
                    full_fppi = np.concatenate([[0.0], fppi_arr])
                    full_mr = np.concatenate([[1.0], 1.0 - np.array(recall)])
                    self.full_curve = {"fppi": full_fppi, "miss_rate": full_mr,
                                        "total_positives": int(npig), "num_images": I0,
                                        "num_predictions_considered": int(len(recall))}

        self.eval = {"params": p, "counts": [T, R, 1, 1], "TP": ys}


def run_subset(cocoGt, preds_official, ht_rng, vis_rng, setup_lbl):
    """
    Runs one subset through the real official pipeline and returns the same
    dict shape your report/plotting code already expects (or None if no
    positive GT for this subset).
    """
    cocoDt = cocoGt.loadRes(copy.deepcopy(preds_official))
    ev = COCOevalWithCurve(cocoGt, cocoDt, "bbox")
    ev.params.imgIds = sorted(cocoGt.getImgIds())
    # allow arbitrary (non-official) subset windows too
    ev.params.HtRng = list(ev.params.HtRng) + [ht_rng]
    ev.params.VisRng = list(ev.params.VisRng) + [vis_rng]
    ev.params.SetupLbl = list(ev.params.SetupLbl) + [setup_lbl]
    custom_id = len(ev.params.HtRng) - 1

    ev.evaluate(custom_id)
    ev.accumulate()

    if ev.full_curve is None:
        return None

    mr = ev.full_curve["miss_rate"]
    sampled = np.maximum(mr[np.clip(
        np.searchsorted(ev.full_curve["fppi"],
                         np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000,
                                   0.1778, 0.3162, 0.5623, 1.0000]),
                         side="right") - 1, 0, None)], 1e-10)
    mr2 = float(np.exp(np.mean(np.log(sampled))))

    return {
        "fppi": ev.full_curve["fppi"],
        "miss_rate": mr,
        "total_positives": ev.full_curve["total_positives"],
        "num_images": ev.full_curve["num_images"],
        "num_predictions_considered": ev.full_curve["num_predictions_considered"],
        "mr2": mr2,
    }