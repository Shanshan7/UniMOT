import motmetrics as mm
import numpy as np 
import os

metrics = list(mm.metrics.motchallenge_metrics)

gt_file="/home/edge/workspace/EDGE/Track/Method_FairMOT/FairMOT/demos/gt.txt"
ts_file="/home/edge/workspace/EDGE/Track/Method_FairMOT/FairMOT/demos/results.txt"
gt=mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
ts=mm.io.loadtxt(ts_file, fmt="mot15-2D")
name=os.path.splitext(os.path.basename(ts_file))[0]

acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=metrics, name=name)
print(mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names))
