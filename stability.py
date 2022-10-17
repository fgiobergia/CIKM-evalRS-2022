import json
import numpy as np
from collections import defaultdict

fname = "flavio.giobergia_polito.it_16653230347834.json"

res_folds = defaultdict(lambda: [])
with open(fname) as f:
    obj = json.load(f)
    res = obj["reclist_reports"]

    for fold in res:
        data = [ m for m in fold["data"] if m["test_name"] != "stats" ]

        scores = {
            m["test_name"]: m["test_result"] if isinstance(m["test_result"], float) else m["test_result"]["mred"] for m in data
        }

        for k, v in scores.items():
            res_folds[k].append(v)
    
    w = []
    for metric, vals in res_folds.items():
        stab = 1 - np.std(vals) / abs(np.mean(vals))
        print(metric, round(stab, 4))
        w.append(stab)
    print("Overall", round(np.mean(w), 4))