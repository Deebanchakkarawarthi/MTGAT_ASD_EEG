import argparse
from pathlib import Path
import yaml, numpy as np, pandas as pd
from mtgat.preprocess import load_and_preprocess, fixed_epochs
from mtgat.connectivity import plv_matrix

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config,encoding="utf-8")); root=Path(cfg["data"]["bids_root"])
    out=Path(cfg["data"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    exts={".bdf",".edf",".fif",".set"}; paths=sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if not paths: raise SystemExit(f"No supported EEG files under {root.resolve()}")
    pp=cfg["preprocessing"]; bands=cfg["features"]["bands"]; rows=[]; feats=[]; adjs=[]
    for path in paths:
        try:
            raw=load_and_preprocess(path,pp["l_freq"],pp["h_freq"],pp["target_sfreq"])
            epochs=fixed_epochs(raw,pp["epoch_seconds"],pp["epoch_overlap"])
            participant=next((x for x in path.parts if x.startswith("sub-")),path.stem)
            for i,e in enumerate(epochs.get_data()):
                adjs.append(plv_matrix(e,raw.info["sfreq"],bands))
                # Baseline feature placeholder: amplitude per channel, replicated across bands.
                amp=np.mean(np.abs(e),axis=-1)
                feats.append(np.stack([amp]*len(bands),axis=0)[:,:,None])
                rows.append({"sample_id":f"{participant}_epoch-{i:05d}","participant_id":participant,"file":str(path),"epoch":i})
            print(f"Processed {path}: {len(epochs)} epochs")
        except Exception as e: print(f"SKIPPED {path}: {e}")
    if not rows: raise SystemExit("No epochs extracted; inspect logs and file formats.")
    np.savez_compressed(out/"epoch_features.npz",adjacency=np.stack(adjs),node_features=np.stack(feats))
    pd.DataFrame(rows).to_csv(out/"epoch_manifest.csv",index=False)
    print(f"Saved {len(rows)} epoch feature rows to {out.resolve()}")
if __name__=="__main__": main()
