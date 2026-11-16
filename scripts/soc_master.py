#!/usr/bin/env python3
# =============================================================================
# soc_master.py — Digital Soil Organic Carbon Mapping
# Study Area : Dhanolti, Uttarakhand
# Model      : Random Forest | Splits: 70:30 & 80:20 | CV: 5-fold, 10-fold, LOOCV
#
# Python port of SOC_MASTER.R (randomForest + caret + terra + sf + ggplot2).
# R behaviour reproduced on purpose, not approximated:
#   set.seed() + sample()          -> RRNG: bit-exact R Mersenne-Twister + "Rejection" sampling
#   caret::createDataPartition     -> create_data_partition: same train/test rows as R, same seed
#   randomForest(nodesize = 5)     -> RForest: bootstrap rows duplicated like R, so nodesize
#                                     counts duplicates; mtry drawn per node; OOB kept
#   rf$importance[, "%IncMSE"]     -> oob_inc_mse: per-tree OOB permutation, unscaled
#   caret::train(metric = "RMSE")  -> tune_mtry: mean fold RMSE, Rsquared = cor^2 (caret style)
#   terra::mask(touches = TRUE)    -> rasterio mask(all_touched = True)
#
# Accuracy upgrade (FINAL_MODEL = "Blend"): mean of (a) a random forest on the selected
# predictors + 6 oblique geographic coordinates (Moller et al. 2020) and (b) an RBF SVR on
# the selected predictors. On 200 R-style 80:20 splits with feature selection redone inside
# each training set, mean test R2 rose from ~0.06 (RF alone) to ~0.14; the blend beat RF in
# ~85% of splits. Picked on seeds 1-100, confirmed on untouched seeds 201-300.
# FINAL_MODEL = "RF" reproduces the R map method exactly. RF_All_Results.csv always
# carries the R (RF-only) numbers plus the blend's numbers on the same splits.
# =============================================================================

import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from joblib import Parallel, delayed, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS
# =============================================================================
SEED        = 42
N_TREE      = 1000          # randomForest(ntree = 1000)
N_TREE_FS   = 500           # feature-selection forest
NODESIZE    = 5             # randomForest default for regression
FINAL_MODEL = "Blend"       # "RF" = R method exactly; "Blend" = mean(RF + oblique coords, SVR)
MODEL_LABEL = {"RF": "Random Forest", "Blend": "RF (oblique coords) + SVR"}
OBLIQUE_ANGLES = (0, 30, 60, 90, 120, 150)   # degrees
N_REPEATS   = 50            # robust evaluation: number of repeated 80:20 splits
PROJECT_DIR = Path(__file__).resolve().parents[1]      # .../SOC_Project
CSV_NAME    = "SOC_COMPLETE_FINAL_3.csv"

DATA_DIR     = PROJECT_DIR / "data"
RASTER_DIR   = DATA_DIR / "rasters"
BOUNDARY_DIR = DATA_DIR / "boundary"
OUT_DIR      = PROJECT_DIR / "output"

RASTER_FILES = {   # predictor -> raster file
    "slope_1": "slope_1.tif", "twi_1": "twi_1.tif", "elevation_1": "elevation_1.tif",
    "aspect_1": "aspect_1.tif", "p_curvature_1": "p_curvature_1.tif",
    "pl_curvature_1": "pl_curvature_1.tif", "mat_celsius": "mat_celsius.tif",
    "map_mm": "map_mm.tif", "precip_seasonality": "precip_seasonality.tif",
    "NDVI": "NDVI.tif", "EVI": "evi.tif", "NDWI": "NDWI.tif", "SAVI": "savi.tif",
    "ndvi_slope": "ndvi_slope.tif", "ndvi_intercept": "ndvi_intercept.tif",
    "ndvi_max": "ndvi_max.tif", "monsoon_ndvi_slope": "monsoon_ndvi_slope.tif",
    "winter_ndvi_slope": "winter_ndvi_slope.tif", "dist_to_stream": "dist_to_stream.tif",
    "lithology": "lithology.tif", "LULC": "LULC.tif", "pH_water": "pH_water.tif",
    "sand_pct": "sand_pct.tif", "clay_pct": "clay_pct.tif", "silt_pct": "silt_pct.tif",
    "nitrogen_gkg": "nitrogen_gkg.tif", "BD_kgdm3": "BD_kgdm3.tif",
}
CATEGORICAL = {"lithology", "LULC"}


# =============================================================================
# R-COMPATIBLE BUILDING BLOCKS
# =============================================================================
class RRNG:
    """R's default RNG, bit-exact: set.seed(seed); Mersenne-Twister; sample.kind = "Rejection"."""

    def __init__(self, seed):
        s = seed & 0xFFFFFFFF
        for _ in range(50):                       # R's initial scrambling
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = []
        for _ in range(625):                      # dummy[0] = mti, dummy[1:] = mt[]
            s = (69069 * s + 1) & 0xFFFFFFFF
            key.append(s)
        self._bg = np.random.MT19937()
        self._bg.state = {"bit_generator": "MT19937",
                          "state": {"key": np.array(key[1:], dtype=np.uint32), "pos": 624}}

    def unif_rand(self):
        x = int(self._bg.random_raw()) * 2.3283064365386963e-10
        if x <= 0.0:
            return 0.5 * 2.328306437080797e-10
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * 2.328306437080797e-10
        return x

    def unif_index(self, dn):
        """R_unif_index(): rejection sampling on random bits."""
        if dn <= 0:
            return 0
        bits = math.ceil(math.log2(dn))
        while True:
            v = 0
            for _ in range(0, bits + 1, 16):
                v = 65536 * v + int(self.unif_rand() * 65536)
            v &= (1 << bits) - 1
            if v < dn:
                return v

    def sample(self, x, size):
        """sample(x, size) without replacement (length(x) > 1)."""
        x, pool, n, out = list(x), list(range(len(x))), len(x), []
        for _ in range(size):
            j = self.unif_index(n)
            out.append(x[pool[j]])
            n -= 1
            pool[j] = pool[n]
        return out


def r_quantile(y, probs):
    """quantile(y, probs, type = 7) with R's exact interpolation formula."""
    x = np.sort(np.asarray(y, float))
    idx = 1 + (len(x) - 1) * np.asarray(probs, float)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    q, h = x[lo - 1].copy(), idx - lo
    m = (idx > lo) & (x[hi - 1] != q)
    q[m] = (1 - h[m]) * q[m] + h[m] * x[hi - 1][m]
    return q


def quantile_groups(y, n_groups):
    """cut(y, unique(quantile(y, seq(0, 1, length = n_groups))), include.lowest = TRUE)."""
    breaks = np.unique(r_quantile(y, np.linspace(0, 1, n_groups)))
    return np.maximum(np.searchsorted(breaks, y, side="left"), 1)


def create_data_partition(y, p, seed, groups=5):
    """caret::createDataPartition(y, p, list = FALSE) after set.seed(seed). 0-based train rows."""
    rng, bins, out = RRNG(seed), quantile_groups(y, max(2, min(groups, len(y)))), []
    for b in np.unique(bins):
        members = np.flatnonzero(bins == b)
        out += list(members) if len(members) == 1 else rng.sample(members, math.ceil(len(members) * p))
    return np.sort(np.array(out))


def _grow_tree(X, y, mtry, nodesize, seed):
    # R splits a node only if it holds more than `nodesize` (bootstrap) rows
    return DecisionTreeRegressor(max_features=mtry, min_samples_split=nodesize + 1,
                                 random_state=seed).fit(X, y)


class RForest:
    """randomForest() regression: ntree bootstrap trees, mtry per node, nodesize, OOB."""

    def __init__(self, mtry, ntree=N_TREE, nodesize=NODESIZE, seed=SEED):
        self.mtry, self.ntree, self.nodesize, self.seed = mtry, ntree, nodesize, seed

    def fit(self, X, y):
        X, y = np.asarray(X, np.float32), np.asarray(y, float)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.inbag_ = rng.integers(0, n, size=(self.ntree, n))
        seeds = rng.integers(0, 2**31 - 1, size=self.ntree)
        mtry = min(self.mtry, X.shape[1])
        self.trees_ = [_grow_tree(X[b], y[b], mtry, self.nodesize, s)
                       for b, s in zip(self.inbag_, seeds)]
        self.oob_ = np.ones((self.ntree, n), bool)
        self.oob_[np.arange(self.ntree)[:, None], self.inbag_] = False
        P, cnt = self.tree_predictions(X), self.oob_.sum(0)
        self.oob_prediction_ = np.where(cnt > 0, (P * self.oob_).sum(0) / np.maximum(cnt, 1), np.nan)
        ok = cnt > 0
        self.oob_mse_ = float(np.mean((y[ok] - self.oob_prediction_[ok]) ** 2))
        self.oob_r2_ = 1 - self.oob_mse_ / np.var(y, ddof=1)       # R: 1 - mse[ntree] / var(y)
        return self

    def tree_predictions(self, X):
        X = np.ascontiguousarray(X, np.float32)
        return np.stack([t.predict(X, check_input=False) for t in self.trees_])

    def predict(self, X):
        return self.tree_predictions(X).mean(0)


def oob_inc_mse(rf, X, y, seed=SEED):
    """rf$importance[, "%IncMSE"]: mean increase in per-tree OOB MSE when the variable is
    permuted (unscaled, exactly what SOC_MASTER.R stores and filters on)."""
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X, np.float32), np.asarray(y, float)
    d = np.zeros((rf.ntree, X.shape[1]))
    for t, tree in enumerate(rf.trees_):
        Xo, yo = X[rf.oob_[t]], y[rf.oob_[t]]
        if len(yo) == 0:
            continue
        base = np.mean((yo - tree.predict(Xo, check_input=False)) ** 2)
        for j in np.unique(tree.tree_.feature[tree.tree_.feature >= 0]):   # unused vars: delta 0
            Xp = Xo.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            d[t, j] = np.mean((yo - tree.predict(Xp, check_input=False)) ** 2) - base
    return d.mean(0)


def node_purity(rf):
    """IncNodePurity: mean total RSS decrease per variable."""
    n = rf.inbag_.shape[1]
    return np.mean([t.tree_.compute_feature_importances(normalize=False) * n
                    for t in rf.trees_], axis=0)


# =============================================================================
# METRICS, CV, TUNING
# =============================================================================
def r2_score(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


def calc_metrics(obs, pred):
    """R calc_metrics(): RMSE, MAE, R2 (1 - SSE/SST), Bias, RPIQ."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    q1, q3 = r_quantile(obs, [0.25, 0.75])
    return {"RMSE": round(rmse, 4), "MAE": round(float(np.mean(np.abs(obs - pred))), 4),
            "R2": round(r2_score(obs, pred), 4), "Bias": round(float(np.mean(pred - obs)), 4),
            "RPIQ": round(float((q3 - q1) / rmse), 4)}


def caret_folds(y, k, seed):
    """trainControl(method = 'cv' / 'LOOCV'): caret::createFolds stratifies on quantile groups."""
    n = len(y)
    if k >= n:
        return [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
    groups = quantile_groups(y, int(min(5, max(2, n // k))))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), groups))


def make_svr():
    """RBF support-vector regression on standardised predictors (blend partner for RF)."""
    return make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))


def oblique(xy):
    """Oblique geographic coordinates: easting/northing projected onto rotated axes.
    Lets RF use location without the blocky artefacts of plain x/y splits."""
    a = np.deg2rad(OBLIQUE_ANGLES)
    return xy[:, :1] * np.cos(a) + xy[:, 1:] * np.sin(a)


def fit_blend(X, y, xy, seed=SEED):
    """Blend = mean of RF(predictors + oblique coords) and SVR(predictors)."""
    Xo = np.c_[X, oblique(xy)]
    return RForest(max(1, int(math.sqrt(Xo.shape[1]))), seed=seed).fit(Xo, y), make_svr().fit(X, y)


def predict_blend(models, X, xy):
    rf, svr = models
    return (rf.predict(np.c_[X, oblique(xy)]) + svr.predict(X)) / 2


def _fit_predict(Xtr, ytr, Xva, mtry, seed):
    return RForest(mtry, seed=seed).fit(Xtr, ytr).predict(Xva)


def tune_mtry(X, y, folds, grid, seed=SEED):
    """caret::train(method = 'rf', tuneGrid = mtry grid, metric = 'RMSE').
    Returns (bestTune mtry, CV_RMSE, CV_R2) as SOC_MASTER.R reports them:
    CV_R2 = max(results$Rsquared) over the grid (caret cor^2), CV_RMSE = RMSE at that row."""
    jobs = [(m, f) for m in grid for f in range(len(folds))]
    preds = Parallel(n_jobs=-1)(
        delayed(_fit_predict)(X[folds[f][0]], y[folds[f][0]], X[folds[f][1]], m, seed + f)
        for m, f in jobs)
    loo = all(len(va) == 1 for _, va in folds)
    rows = []
    for m in grid:
        fold_preds = [p for (mm, _), p in zip(jobs, preds) if mm == m]
        if loo:   # caret pools LOOCV hold-outs
            oof = np.concatenate(fold_preds)
            obs = np.concatenate([y[va] for _, va in folds])
            rows.append((m, np.sqrt(np.mean((obs - oof) ** 2)), np.corrcoef(obs, oof)[0, 1] ** 2))
        else:
            r = [(np.sqrt(np.mean((y[va] - p) ** 2)), np.corrcoef(y[va], p)[0, 1] ** 2)
                 for (_, va), p in zip(folds, fold_preds)]
            rows.append((m, np.mean([a for a, _ in r]), np.nanmean([b for _, b in r])))
    best = min(rows, key=lambda r: r[1])          # bestTune: lowest RMSE, ties -> first
    top = max(rows, key=lambda r: r[2])           # R: max(Rsquared) and its RMSE
    return best[0], float(top[1]), float(top[2])


# =============================================================================
# PIPELINE
# =============================================================================
def load_clean_data():
    print("[ 1/6 ] LOADING & CLEANING DATA")
    print("-" * 45)
    df_raw = pd.read_csv(DATA_DIR / CSV_NAME)
    print(f"Raw data: {df_raw.shape[0]} rows | {df_raw.shape[1]} cols")

    drop_names = {"sno", "id", "fid", "lat", "long", "lon", "x", "y",
                  "longitude", "latitude", "objectid", "geometry"}
    drop_cols = [c for c in df_raw.columns if c.strip().lower() in drop_names]
    df = df_raw.drop(columns=drop_cols)
    print("Dropped:", ", ".join(drop_cols) or "(none)")

    soc_col = next((c for c in df.columns if c.strip().lower() in {"soc", "soc_pct", "oc"}), None)
    if soc_col is None:
        raise SystemExit("SOC column not found. Check CSV.")
    df = df.rename(columns={soc_col: "soc"})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["soc"].notna() & (df["soc"] > 0) & (df["soc"] < 25)].copy()

    pred_cols = [c for c in df.columns if c != "soc"]
    nzv = [c for c in pred_cols if df[c].dropna().nunique() <= 2 or df[c].std(skipna=True) < 1e-8]
    if nzv:
        print("Removing NZV:", ", ".join(nzv))
        df = df.drop(columns=nzv)
    pred_cols = [c for c in df.columns if c != "soc"]

    q1, q3 = r_quantile(df["soc"], [0.25, 0.75])           # 3 x IQR on SOC
    iq = q3 - q1
    df = df[(df["soc"] >= q1 - 3 * iq) & (df["soc"] <= q3 + 3 * iq)].copy()
    for c in pred_cols:
        df[c] = df[c].fillna(df[c].median())

    feats = [f for f in RASTER_FILES if f in df.columns]
    model_df = df[["soc"] + feats].dropna()
    lon = next(c for c in df_raw.columns if c.strip().lower() in {"long", "lon", "longitude"})
    lat = next(c for c in df_raw.columns if c.strip().lower() in {"lat", "latitude"})
    xy = sample_xy(df_raw.loc[model_df.index, lon].values, df_raw.loc[model_df.index, lat].values)
    model_df = model_df.reset_index(drop=True)
    print(f"Clean dataset: {len(model_df)} samples | {len(feats)} predictors with rasters")
    print(f"SOC: mean={model_df['soc'].mean():.2f} | sd={model_df['soc'].std():.2f} | "
          f"min={model_df['soc'].min():.2f} | max={model_df['soc'].max():.2f}\n")
    model_df.to_csv(OUT_DIR / "data_clean.csv", index=False)
    return model_df, feats, xy


def sample_xy(lon, lat):
    """Sample lon/lat (WGS84) -> metres in the reference raster's CRS (UTM 44N)."""
    _use_bundled_gis_data()
    import rasterio
    from pyproj import Transformer
    with rasterio.open(RASTER_DIR / "slope_1.tif") as ref:
        crs = ref.crs.to_wkt()
    return np.c_[Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)]


def rank_features(df, feats):
    """500-tree RF (mtry = floor(sqrt(p))) on all predictors, ranked by OOB %IncMSE."""
    X, y = df[feats].values, df["soc"].values
    rf = RForest(max(1, int(math.sqrt(len(feats)))), ntree=N_TREE_FS).fit(X, y)
    return (pd.DataFrame({"Variable": feats, "IncMSE": oob_inc_mse(rf, X, y),
                          "NodePurity": node_purity(rf)})
            .sort_values("IncMSE", ascending=False).reset_index(drop=True))


def select_features(model_df, feats):
    print("[ 2/6 ] FEATURE SELECTION — RF %IncMSE")
    print("-" * 45)
    imp_df = rank_features(model_df, feats)
    print(imp_df.round(4).to_string(index=False))
    imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    sel = imp_df.loc[imp_df["IncMSE"] > 0, "Variable"].tolist()
    print(f"\nSelected {len(sel)} features (%IncMSE > 0)\n")

    plot_imp = imp_df[imp_df["IncMSE"] > 0].sort_values("IncMSE")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_imp["Variable"], plot_imp["IncMSE"],
            color=plt.cm.plasma(np.linspace(0.1, 0.95, len(plot_imp))), height=0.7)
    ax.set_xlabel("%IncMSE")
    ax.set_title("RF Variable Importance — %IncMSE", fontweight="bold", loc="left")
    ax.text(0, 1.02, f"SOC Prediction | Dhanolti | {len(sel)} selected features",
            transform=ax.transAxes, fontsize=10, color="gray")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Plot_Feature_Importance.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return sel


def train_models(model_df, sel, xy):
    print("[ 3/6 ] MODEL TRAINING — RF with 70:30 & 80:20 | 5-fold, 10-fold, LOOCV")
    print("-" * 45)
    p = len(sel)
    grid = sorted({max(1, int(math.sqrt(p) / 2)), max(1, int(math.sqrt(p))),
                   max(1, int(math.sqrt(p) * 1.5)), max(1, p // 3)})
    print("mtry grid:", grid)
    X, y = model_df[sel].values, model_df["soc"].values

    results, holdout = [], {}
    for sp_name, frac in {"70:30": 0.70, "80:20": 0.80}.items():
        tr = create_data_partition(y, frac, SEED)
        te = np.setdiff1d(np.arange(len(y)), tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        print(f"\n{sp_name} | Train: {len(tr)} | Test: {len(te)}")

        base = RForest(max(1, int(math.sqrt(p)))).fit(Xtr, ytr)
        pte, ptr = base.predict(Xte), base.predict(Xtr)
        m_te, m_tr = calc_metrics(yte, pte), calc_metrics(ytr, ptr)
        m_bl = calc_metrics(yte, predict_blend(fit_blend(Xtr, ytr, xy[tr]), Xte, xy[te]))
        print(f"  Holdout Test  R2={m_te['R2']:.3f} RMSE={m_te['RMSE']:.3f}"
              f"   | Blend R2={m_bl['R2']:.3f} RMSE={m_bl['RMSE']:.3f}")
        print(f"  Holdout Train R2={m_tr['R2']:.3f} RMSE={m_tr['RMSE']:.3f}")
        holdout[sp_name] = dict(obs_test=yte, pred_test=pte, obs_train=ytr, pred_train=ptr)

        for cv_name, k in {"5-fold": 5, "10-fold": 10, "LOOCV": len(tr)}.items():
            print(f"  CV: {cv_name:<8} ... ", end="", flush=True)
            mtry, cv_rmse, cv_r2 = tune_mtry(Xtr, ytr, caret_folds(ytr, k, SEED), grid)
            m = calc_metrics(yte, RForest(mtry).fit(Xtr, ytr).predict(Xte))
            results.append({"Split": sp_name, "CV": cv_name, "Best_mtry": mtry,
                            "Train_R2": round(m_tr["R2"], 3), "Train_RMSE": round(m_tr["RMSE"], 3),
                            "Test_R2": round(m["R2"], 3), "Test_RMSE": round(m["RMSE"], 3),
                            "Test_MAE": round(m["MAE"], 3), "Test_RPIQ": round(m["RPIQ"], 3),
                            "CV_R2": round(cv_r2, 3), "CV_RMSE": round(cv_rmse, 3),
                            # blend is not mtry-tuned: same value for every CV row of a split
                            "Blend_Test_R2": round(m_bl["R2"], 3), "Blend_Test_RMSE": round(m_bl["RMSE"], 3)})
            print(f"mtry={mtry} | Test R2={m['R2']:.3f} | CV R2={cv_r2:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "RF_All_Results.csv", index=False)
    print("\n=== ALL MODEL RESULTS ===")
    print(res.sort_values("Test_R2", ascending=False).to_string(index=False))
    plot_results(res, holdout)
    return res


def make_scatter(ax, obs, pred, title_txt):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ax.scatter(obs, pred, alpha=0.7, color="#2166AC", s=35, zorder=3)
    lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1.2, zorder=2)
    z = np.polyfit(obs, pred, 1)
    xs = np.linspace(obs.min(), obs.max(), 100)
    ys = np.polyval(z, xs)
    se = np.std(pred - np.polyval(z, obs)) * 1.96
    ax.plot(xs, ys, color="#4DAC26", lw=1.6, zorder=4)
    ax.fill_between(xs, ys - se, ys + se, color="#B8E186", alpha=0.3, zorder=1)
    ax.text(0.03, 0.97, f"R² = {r2_score(obs, pred):.3f}\nRMSE = {np.sqrt(np.mean((obs - pred) ** 2)):.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11, style="italic")
    ax.set_title(title_txt, fontweight="bold", fontsize=11, loc="left")
    ax.set_xlabel("Observed SOC (%)")
    ax.set_ylabel("Predicted SOC (%)")
    ax.grid(alpha=0.3)


def plot_results(res, holdout):
    ho70, ho80 = holdout["70:30"], holdout["80:20"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    make_scatter(axes[0, 0], ho80["obs_test"], ho80["pred_test"], "80:20 — Test Set")
    make_scatter(axes[0, 1], ho70["obs_test"], ho70["pred_test"], "70:30 — Test Set")
    make_scatter(axes[1, 0], ho80["obs_train"], ho80["pred_train"], "80:20 — Training Set")
    make_scatter(axes[1, 1], ho70["obs_train"], ho70["pred_train"], "70:30 — Training Set")
    fig.suptitle("SOC Random Forest — Observed vs Predicted", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Plot_ObsVsPred.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    cv_order, w = ["5-fold", "10-fold", "LOOCV"], 0.35
    xs = np.arange(len(cv_order))
    for i, (sp, col) in enumerate({"70:30": "#66C2A5", "80:20": "#FC8D62"}.items()):
        sub = res[res["Split"] == sp].set_index("CV")
        vals = [sub.loc[c, "Test_R2"] for c in cv_order]
        pos = xs + (i - 0.5) * w
        ax.bar(pos, vals, width=w, label=sp, color=col, edgecolor="black", linewidth=0.4)
        for px, v in zip(pos, vals):
            ax.text(px, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(cv_order)
    ax.set_ylabel("Test R²")
    ax.set_title("RF Test R² — Splits & CV Methods", fontweight="bold", loc="left")
    ax.legend(title="Split")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Plot_Model_Comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _robust_one(df, feats, xy, seed):
    y = df["soc"].values
    tr = create_data_partition(y, 0.8, seed)
    te = np.setdiff1d(np.arange(len(y)), tr)
    imp = rank_features(df.iloc[tr], feats)               # selection sees training rows only
    sel = imp.loc[imp["IncMSE"] > 0, "Variable"].tolist() or imp["Variable"].tolist()[:5]
    Xtr, Xte = df.iloc[tr][sel].values, df.iloc[te][sel].values
    rf = RForest(max(1, int(math.sqrt(len(sel)))), seed=seed).fit(Xtr, y[tr]).predict(Xte)
    bl = predict_blend(fit_blend(Xtr, y[tr], xy[tr], seed), Xte, xy[te])
    return seed, r2_score(y[te], rf), r2_score(y[te], bl)


def robust_eval(model_df, feats, xy):
    """One 16-sample test set is mostly luck. Average over N_REPEATS R-style 80:20 splits,
    redoing feature selection inside each training set so the test rows never leak in."""
    print(f"\n[ 4b ] ROBUST CHECK — {N_REPEATS} R-style 80:20 splits, features re-selected per split")
    print("-" * 45)
    rows = Parallel(n_jobs=-1)(delayed(_robust_one)(model_df, feats, xy, s) for s in range(1, N_REPEATS + 1))
    rob = pd.DataFrame(rows, columns=["Seed", "RF_Test_R2", "Blend_Test_R2"])
    rob.to_csv(OUT_DIR / "Robust_Eval.csv", index=False)
    d = rob["Blend_Test_R2"] - rob["RF_Test_R2"]
    print(f"Mean test R2  RF: {rob['RF_Test_R2'].mean():.3f} | Blend: {rob['Blend_Test_R2'].mean():.3f} | "
          f"gain {d.mean():+.3f} ± {d.std() / math.sqrt(len(d)):.3f} (SE) | blend wins {np.mean(d > 0):.0%} of splits")
    return rob


def _use_bundled_gis_data():
    """PostGIS & co. export PROJ_LIB/GDAL_DATA for their own older PROJ, which breaks pip
    rasterio/pyogrio wheels ("proj.db ... DATABASE.LAYOUT.VERSION"). Wheels ship their own data."""
    import importlib.util
    if (Path(importlib.util.find_spec("rasterio").origin).parent / "proj_data").is_dir():
        for var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
            os.environ.pop(var, None)


def map_soc(model_df, sel, best_row, rob, xy):
    _use_bundled_gis_data()
    import rasterio
    import geopandas as gpd
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    print("\n[ 5/6 ] SPATIAL PREDICTION & MAPPING")
    print("-" * 45)
    with rasterio.open(RASTER_DIR / "slope_1.tif") as ref:
        prof, tr, crs, (h, w), res = ref.profile.copy(), ref.transform, ref.crs, ref.shape, ref.res
    print(f"Reference: {crs.to_string()} | {res[0]:.0f} m | {h} x {w} px")

    def load_align(path, categorical):
        with rasterio.open(path) as src:
            dst = np.full((h, w), np.nan, dtype="float32")
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs or crs,
                      dst_transform=tr, dst_crs=crs, src_nodata=src.nodata, dst_nodata=np.nan,
                      resampling=Resampling.nearest if categorical else Resampling.bilinear)
        return dst

    layers, missing = {}, []
    for f in sel:
        path = RASTER_DIR / RASTER_FILES[f]
        arr = load_align(path, f in CATEGORICAL) if path.exists() else None
        if arr is None or not np.isfinite(arr).any():
            missing.append(f)
        else:
            layers[f] = arr
    map_feats = [f for f in sel if f not in missing]
    X_all, y_all = model_df[map_feats].values, model_df["soc"].values
    mtry = int(best_row["Best_mtry"]) if not missing else max(1, int(math.sqrt(len(map_feats))))
    if missing:
        print("Missing/empty rasters:", ", ".join(missing), "-> retraining without them")

    print(f"Final RF on ALL {len(y_all)} samples | mtry={mtry} | {len(map_feats)} predictors")
    final_rf = RForest(mtry).fit(X_all, y_all)
    oob_rmse = float(np.sqrt(np.nanmean((y_all - final_rf.oob_prediction_) ** 2)))
    print(f"Final RF OOB R2={final_rf.oob_r2_:.3f} | OOB RMSE={oob_rmse:.3f}")
    blend = fit_blend(X_all, y_all, xy) if FINAL_MODEL == "Blend" else None
    dump({"model": final_rf, "blend_rf_svr": blend, "final_model": FINAL_MODEL,
          "features": map_feats, "oblique_angles": OBLIQUE_ANGLES},
         OUT_DIR / "SOC_FinalRF_Model.joblib")

    flat = np.stack([layers[f] for f in map_feats]).reshape(len(map_feats), -1).T
    valid = np.isfinite(flat).all(axis=1)
    Xv = np.ascontiguousarray(flat[valid], np.float32)
    rows, cols = np.divmod(np.flatnonzero(valid), w)               # pixel-centre coordinates
    pix_xy = np.c_[tr.c + (cols + 0.5) * tr.a, tr.f + (rows + 0.5) * tr.e]
    map_rf, Xpix = (final_rf, Xv) if blend is None else (blend[0], np.c_[Xv, oblique(pix_xy)])
    Xpix = np.ascontiguousarray(Xpix, np.float32)
    s = np.zeros(len(Xv))
    ss = np.zeros(len(Xv))
    for t in map_rf.trees_:                      # running sums: no (ntree x pixels) matrix
        pt = t.predict(Xpix, check_input=False)
        s += pt
        ss += pt * pt
    T = map_rf.ntree
    mean = s / T
    sd = np.sqrt(np.maximum(ss - T * mean ** 2, 0) / (T - 1))    # R sd() across trees
    if blend is not None:
        mean = (mean + blend[1].predict(Xv)) / 2
    soc_map = np.full(h * w, np.nan, np.float32)
    sd_map = np.full(h * w, np.nan, np.float32)
    soc_map[valid], sd_map[valid] = mean, sd
    soc_map, sd_map = soc_map.reshape(h, w), sd_map.reshape(h, w)

    shp = sorted(BOUNDARY_DIR.glob("*.shp"))[0]
    bnd = gpd.read_file(shp)
    bnd = bnd.set_crs(crs) if bnd.crs is None else bnd.to_crs(crs)
    geom = bnd.union_all()

    def mask_write(arr, name):
        p = prof.copy()
        p.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
        with rasterio.MemoryFile() as mf:
            with mf.open(**p) as ds:
                ds.write(arr.astype("float32"), 1)
                out, otr = rio_mask(ds, [geom], crop=True, all_touched=True, nodata=np.nan)
        p.update(height=out.shape[1], width=out.shape[2], transform=otr)
        with rasterio.open(OUT_DIR / name, "w", **p) as dst:
            dst.write(out[0], 1)
        return out[0], otr, p

    soc_m, otr, oprof = mask_write(soc_map, "SOC_Predicted.tif")
    sd_m, _, _ = mask_write(sd_map, "SOC_Uncertainty.tif")

    zones = np.digitize(soc_m, [1, 2, 3, 4], right=True).astype("float32") + 1   # (..1],(1..2],...
    zones[np.isnan(soc_m)] = np.nan
    with rasterio.open(OUT_DIR / "SOC_Zones.tif", "w",
                       **{**oprof, "dtype": "uint8", "nodata": 255}) as dst:   # R: INT1U
        dst.write(np.nan_to_num(zones, nan=255).astype("uint8"), 1)

    vals = soc_m[np.isfinite(soc_m)]
    area_ha = len(vals) * res[0] * res[1] / 10000
    stats = {"Area (ha)": round(area_ha, 1), "Min SOC (%)": round(float(vals.min()), 3),
             "Max SOC (%)": round(float(vals.max()), 3), "Mean SOC (%)": round(float(vals.mean()), 3),
             "Median SOC (%)": round(float(np.median(vals)), 3),
             "Std Dev (%)": round(float(vals.std(ddof=1)), 3),
             "Mean Uncertainty SD (%)": round(float(np.nanmean(sd_m)), 3),
             "OOB R2 (RF)": round(final_rf.oob_r2_, 3), "OOB RMSE (RF)": round(oob_rmse, 3),
             "Final Model": FINAL_MODEL,
             "Best Split": best_row["Split"], "Best CV": best_row["CV"], "Best mtry": best_row["Best_mtry"],
             "Test R2 (RF)": best_row["Test_R2"], "Test RMSE (RF)": best_row["Test_RMSE"],
             "Test R2 (Blend)": best_row["Blend_Test_R2"], "Test RMSE (Blend)": best_row["Blend_Test_RMSE"],
             f"Mean Test R2 RF ({N_REPEATS} splits)": round(rob["RF_Test_R2"].mean(), 3),
             f"Mean Test R2 Blend ({N_REPEATS} splits)": round(rob["Blend_Test_R2"].mean(), 3),
             "Variables Used": "; ".join(map_feats), "N Samples": len(y_all)}
    pd.DataFrame({"Metric": list(stats), "Value": list(stats.values())}).to_csv(
        OUT_DIR / "SOC_Map_Summary.csv", index=False)
    print("\n" + "═" * 50)
    for k in list(stats)[:-2]:
        print(f"  {k:<30}: {stats[k]}")
    print("═" * 50)

    print("\n[ 6/6 ] MAP VISUALISATION")
    print("-" * 45)
    hm, wm = soc_m.shape
    extent = (otr.c, otr.c + wm * otr.a, otr.f + hm * otr.e, otr.f)
    bplot = gpd.GeoDataFrame(geometry=[geom], crs=crs)

    def save_map(arr, fname, title, sub=None, cbar=None, **imkw):
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(arr, extent=extent, origin="upper", **imkw)
        bplot.boundary.plot(ax=ax, color="black", linewidth=1.0)
        if cbar:
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label(cbar)
        else:
            ax.legend(handles=[Patch(facecolor=c, edgecolor="black", label=l)
                               for c, l in zip(zone_colors, zone_labels)],
                      title="SOC Class", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        ax.set_title(title, fontweight="bold", fontsize=13, loc="left")
        if sub:
            ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color="gray")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        fig.text(0.99, 0.01, f"SCORPAN-E Framework | {MODEL_LABEL[FINAL_MODEL]} | 30m resolution",
                 ha="right", fontsize=8, color="gray")
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    zone_labels = ["Very Low <1%", "Low 1-2%", "Medium 2-3%", "High 3-4%", "Very High >4%"]
    zone_colors = ["#8B1A1A", "#D2691E", "#F4A460", "#90EE90", "#228B22"]
    save_map(soc_m, "SOC_Map.png", "Predicted Soil Organic Carbon — Dhanolti, Uttarakhand",
             sub=(f"{MODEL_LABEL[FINAL_MODEL]} (ntree={N_TREE}) | Mean={vals.mean():.2f}% | "
                  f"Area={area_ha:.0f} ha | n={len(y_all)} | Best: {best_row['Split']}/{best_row['CV']} | "
                  f"Test R²={best_row['Blend_Test_R2' if FINAL_MODEL == 'Blend' else 'Test_R2']:.3f}"),
             cbar="SOC (%)", cmap="viridis", interpolation="bilinear")
    save_map(zones, "SOC_Zone_Map.png", "SOC Zone Map — Dhanolti, Uttarakhand",
             cmap=ListedColormap(zone_colors), interpolation="nearest",
             norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], len(zone_colors)))
    save_map(sd_m, "SOC_Uncertainty_Map.png", "SOC Prediction Uncertainty — Dhanolti",
             sub=f"Standard Deviation across {N_TREE} RF Trees", cbar="SD (%)",
             cmap="YlOrRd", interpolation="bilinear")


def main():
    os.chdir(PROJECT_DIR)
    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 65)
    print("  SOC DIGITAL SOIL MAPPING — DHANOLTI, UTTARAKHAND")
    print("=" * 65, "\n")

    model_df, feats, xy = load_clean_data()
    sel = select_features(model_df, feats)
    res = train_models(model_df[["soc"] + sel], sel, xy)

    print("\n[ 4/6 ] SELECTING BEST MODEL")
    print("-" * 45)
    best = res.sort_values(["Test_R2", "CV_R2"], ascending=False).iloc[0].to_dict()
    print(f"BEST: Split={best['Split']} | CV={best['CV']} | mtry={best['Best_mtry']} | "
          f"Test R2={best['Test_R2']:.3f} | CV R2={best['CV_R2']:.3f} | "
          f"Blend Test R2={best['Blend_Test_R2']:.3f}")

    rob = robust_eval(model_df, feats, xy)
    map_soc(model_df, sel, best, rob, xy)
    print("\nPIPELINE COMPLETE — outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
