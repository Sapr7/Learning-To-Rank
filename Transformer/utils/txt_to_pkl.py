import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import fire

def txt_to_pkl(txt_path, pkl_path, n_features=None):
    if n_features is None:
        max_feat = 0
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()[2:]
                for feat in parts:
                    k = int(feat.split(":")[0])
                    max_feat = max(max_feat, k)
        n_features = max_feat
        print("Detected features:", n_features)

    data = defaultdict(list)

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            label = float(parts[0])
            qid = int(parts[1].split(":")[1])

            x = np.zeros(n_features, dtype=np.float32)

            for feat in parts[2:]:
                k, v = feat.split(":")
                k = int(k) - 1
                x[k] = float(v)

            data[qid].append((x, label))

    rows = []
    for qid, docs in data.items():
        docs = sorted(docs, key=lambda x: -x[1])
        X = np.vstack([d[0] for d in docs])
        y = np.array([d[1] for d in docs])
        rows.append({
            "query_id": qid,
            "fl_features": X,
            "labels": y,
            "doc_id": list(range(len(docs)))
        })

    df = pd.DataFrame(rows)

    with open(pkl_path, "wb") as f:
        pickle.dump(df.to_dict(orient="list"), f)

    print("Saved:", pkl_path)

if __name__ == "__main__":
    fire.Fire(txt_to_pkl)