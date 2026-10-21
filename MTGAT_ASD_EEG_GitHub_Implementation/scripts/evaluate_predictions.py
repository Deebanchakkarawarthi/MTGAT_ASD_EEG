import argparse,json,pandas as pd
from mtgat.metrics import classification_metrics
p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); a=p.parse_args()
d=pd.read_csv(a.predictions)
print(json.dumps(classification_metrics(d.y_true.to_numpy(),d[["p0","p1"]].to_numpy()),indent=2))
