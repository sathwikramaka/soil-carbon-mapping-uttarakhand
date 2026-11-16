#!/usr/bin/env python3
"""Spatial validation for soc_master.py (run soc_master.py first: needs output/SOC_Predicted.tif).

1. Random 5-fold vs spatially blocked 5-fold CV (blocks = k-means clusters of sample
   coordinates), 20 repetitions, feature selection redone inside every training fold.
2. Which CV matches the mapping task? Compares nearest-neighbour distances:
   map pixel -> nearest sample  vs  CV test sample -> nearest training sample
   (Mila et al. 2022, Methods Ecol. Evol.). The CV whose distances match the map's is the
   one that estimates map accuracy.
Writes output/Spatial_Validation.csv and output/Spatial_Validation_Distances.csv."""
import contextlib
import io
import math

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

import soc_master as sm

N_REP, K = 20, 5


def _fold(df, feats, xy, tr, te, seed):
    y = df["soc"].values
    imp = sm.rank_features(df.iloc[tr], feats)
    sel = imp.loc[imp["IncMSE"] > 0, "Variable"].tolist() or imp["Variable"].tolist()[:5]
    Xtr, Xte = df.iloc[tr][sel].values, df.iloc[te][sel].values
    rf = sm.RForest(max(1, int(math.sqrt(len(sel)))), seed=seed).fit(Xtr, y[tr]).predict(Xte)
    sv = sm.make_svr().fit(Xtr, y[tr]).predict(Xte)
    bl = sm.predict_blend(sm.fit_blend(Xtr, y[tr], xy[tr], seed), Xte, xy[te])
    return te, {"RF (R method)": rf, "RF+SVR": (rf + sv) / 2, "Blend": bl}


def folds(scheme, xy, rep):
    if scheme == "Spatial blocks":
        return KMeans(K, n_init=10, random_state=rep).fit_predict(xy)
    return np.random.default_rng(rep).permutation(np.arange(len(xy)) % K)


def run():
    with contextlib.redirect_stdout(io.StringIO()):
        df, feats, xy = sm.load_clean_data()
    y = df["soc"].values
    rows, dist = [], {}
    for scheme in ("Random", "Spatial blocks"):
        d = []
        for rep in range(N_REP):
            lab = folds(scheme, xy, rep)
            for b in range(K):
                d.append(cKDTree(xy[lab != b]).query(xy[lab == b])[0])
            outs = Parallel(n_jobs=-1)(
                delayed(_fold)(df, feats, xy, np.flatnonzero(lab != b), np.flatnonzero(lab == b), rep * 10 + b)
                for b in range(K))
            pred = {m: np.zeros(len(y)) for m in outs[0][1]}
            for te, o in outs:
                for m, v in o.items():
                    pred[m][te] = v
            rows += [dict(CV=f"{scheme} {K}-fold", Rep=rep, Model=m, R2=sm.r2_score(y, v),
                          RMSE=float(np.sqrt(np.mean((y - v) ** 2)))) for m, v in pred.items()]
        dist[f"CV test -> train ({scheme})"] = np.concatenate(d)

    sm._use_bundled_gis_data()
    import rasterio
    with rasterio.open(sm.OUT_DIR / "SOC_Predicted.tif") as r:
        a, t = r.read(1), r.transform
    rr, cc = np.nonzero(np.isfinite(a))
    pix = np.c_[t.c + (cc + .5) * t.a, t.f + (rr + .5) * t.e]
    dist["Map pixel -> sample"] = cKDTree(xy).query(pix)[0]

    res = pd.DataFrame(rows)
    summ = res.groupby(["CV", "Model"]).agg(Mean_R2=("R2", "mean"), SD_R2=("R2", "std"),
                                            Mean_RMSE=("RMSE", "mean")).round(3).reset_index()
    q = [10, 25, 50, 75, 90]
    dsum = pd.DataFrame({k: np.percentile(v, q).round(0) for k, v in dist.items()},
                        index=[f"P{p} (m)" for p in q]).T
    summ.to_csv(sm.OUT_DIR / "Spatial_Validation.csv", index=False)
    dsum.to_csv(sm.OUT_DIR / "Spatial_Validation_Distances.csv")
    return summ, dsum


if __name__ == "__main__":
    summ, dsum = run()
    print(summ.to_string(index=False))
    print("\nNearest-neighbour distances (the CV that matches the map row estimates map accuracy):")
    print(dsum.to_string())
