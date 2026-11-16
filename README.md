<div align="center">

# 🏔️ Soil Organic Carbon Mapping — Dhanolti, Uttarakhand

**How much organic carbon do the topsoils of a Himalayan watershed hold, and where?**
Digital soil mapping at 30 m with Random Forest and a Random Forest + SVR blend on 27 Earth Engine covariates, structured on the SCORPAN-E framework and checked with spatial validation.

[![SOC map](https://img.shields.io/badge/SOC_map-View-00d4ff?style=for-the-badge&logo=googleearth&logoColor=white)](output/SOC_Map.png)
[![Pipeline](https://img.shields.io/badge/R_pipeline-Read-00ff88?style=for-the-badge&logo=r&logoColor=white)](SOC_MASTER.R)

![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)
![Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-4285F4?logo=google&logoColor=white)
![randomForest](https://img.shields.io/badge/randomForest-2E7D32)
![caret](https://img.shields.io/badge/caret-6A1B9A)
![e1071](https://img.shields.io/badge/e1071_SVR-C62828)
![terra](https://img.shields.io/badge/terra_+_sf-8D6E63)
![QGIS](https://img.shields.io/badge/QGIS-589632?logo=qgis&logoColor=white)

<img src="output/SOC_Map.png" width="760" alt="Predicted soil organic carbon map, Dhanolti">

</div>

---

## 📊 Key results

| | Random Forest | Blend (RF + SVR) |
|---|---|---|
| Best single-split test R² | **0.438** (80:20 split, LOOCV-tuned) | — |
| Holdout test R² — 80:20 / 70:30 | 0.414 / 0.347 | 0.416 / 0.257 |
| Training R² — 80:20 / 70:30 | 0.824 / 0.797 | — |
| **Mean test R², 50 repeated 80:20 splits** | **0.061** | **0.130** |
| Mean R², random 5-fold CV (×20) | 0.046 | 0.115 |
| Mean R², spatially blocked 5-fold CV (×20) | −0.318 | −0.320 |

| | |
|---|---|
| 🗺️ Map (blend) | mean SOC 3.12 %, range 2.68–3.42 %, 11,789 ha at 30 m |
| 🌲 Models | Random Forest, 1,000 trees · blend = mean of RF (covariates + 6 oblique geographic coordinates) and radial SVR |
| 🧭 Covariates | 19 of 27 kept by RF %IncMSE; top five: NDVI intercept, distance to stream, NDVI max, precipitation seasonality, mean annual temperature |
| 🎯 Sampling | 80 field samples · 70:30 and 80:20 splits · 5-fold, 10-fold and LOOCV tuning |

> ⚠️ **Read the 0.438 carefully.** It comes from one favourable 16-sample test set, and covariate selection saw those test rows. Over 50 repeated splits, with selection redone inside each training set, the Random Forest averages **R² ≈ 0.06** and the blend **≈ 0.13**. The blend beats the Random Forest in 76 % of those splits.

## 🧪 Validation: which number describes the map?

Spatially blocked cross-validation holds out whole regions (k-means clusters of sample locations). Whether that or random cross-validation describes the map depends on how far map pixels are from the samples (Milà et al. 2022):

| Nearest-neighbour distance | Median | 90th percentile |
|---|---|---|
| Map pixel → nearest sample | 605 m | 1.33 km |
| Random CV: test → training sample | 687 m | 1.46 km |
| Spatial-block CV: test → training sample | 2.43 km | 4.51 km |

- **Inside the watershed**, map pixels are as close to samples as random-CV test points are, so random CV is the right accuracy estimate for this map: **R² ≈ 0.05 (Random Forest) and ≈ 0.12 (blend)**.
- **Outside it**, every model scores **R² < 0**. The covariates do not transfer to unsampled regions, so the model should not be applied beyond the sampled area.

## 🔍 What it does

- **Covariates from Earth Engine** — terrain, climate, vegetation dynamics, hydrology, lithology, land use and soil properties for every pixel.
- **Feature selection by importance** — Random Forest %IncMSE keeps the covariates that improve out-of-bag prediction.
- **Splits × cross-validation** — Random Forest on 70:30 and 80:20 splits, `mtry` tuned by 5-fold, 10-fold and leave-one-out CV.
- **Improved model** — a blend of a Random Forest with oblique geographic coordinates and a support-vector regression roughly doubles accuracy on held-out samples. Set `FINAL_MODEL <- "RF"` to map with the Random Forest alone.
- **Honest validation** — repeated splits without selection leakage, plus random vs spatially blocked CV with a distance check.
- **Uncertainty as a map** — the spread across 1,000 trees is exported next to the prediction, and the map is classified into five SOC zones.

## 🛰️ Covariates

| Category | Variables |
|----------|-----------|
| **Terrain** | Slope, TWI, Elevation, Aspect, Plan Curvature, Profile Curvature |
| **Climate** | Mean Annual Temperature, Mean Annual Precipitation, Precipitation Seasonality |
| **Vegetation** | NDVI, EVI, NDWI, SAVI, NDVI Slope, NDVI Intercept, NDVI Max, Monsoon NDVI Slope, Winter NDVI Slope |
| **Hydrology** | Distance to Stream |
| **Land** | Lithology, LULC |
| **Soil Properties** | pH, Sand %, Clay %, Silt %, Nitrogen (g/kg), Bulk Density (kg/dm³) |

## 🚀 Run it yourself

Put the data in `data/` next to `SOC_MASTER.R` (layout below), then from the repository folder:

```bash
Rscript SOC_MASTER.R
```

or `source("SOC_MASTER.R")` in RStudio. Missing packages are installed on first run. The full pipeline, including the repeated-split and spatial checks, takes about a minute on a multi-core PC. Settings such as `FINAL_MODEL`, `RUN_ROBUST` and `RUN_SPATIAL_CV` are at the top of the script.

<details>
<summary><b>Pipeline, section by section</b></summary>

| § | What it does |
|---|---|
| 1 | Load and clean: ID columns, SOC range, near-zero variance, 3×IQR outliers, median imputation |
| 2 | Covariate selection by Random Forest %IncMSE |
| 3 | Random Forest on 70:30 and 80:20 splits; `mtry` tuned by 5-fold, 10-fold, LOOCV; blend on the same rows |
| 4 | Best model and final fit on all samples; 4b: 50 repeated splits with selection inside training |
| 5 | Spatial prediction → SOC map, uncertainty map (SD across trees), zone map |
| 6 | Spatial validation: random vs spatially blocked CV and the distance check |

</details>

<details>
<summary><b>Required input layout</b></summary>

```
SOC_MASTER.R
data/
  SOC_COMPLETE_FINAL_3.csv     field samples (SOC, lat/long, covariate values)
  rasters/                     27 covariate GeoTIFFs
  boundary/                    study-area shapefile
output/                        created by the script
```

</details>

<details>
<summary><b>Outputs written to <code>output/</code></b></summary>

| File | Description |
|------|-------------|
| `SOC_Predicted.tif` · `SOC_Uncertainty.tif` · `SOC_Zones.tif` | Prediction, SD across trees, five-class zones (30 m) |
| `SOC_Map.png` · `SOC_Zone_Map.png` · `SOC_Uncertainty_Map.png` | Maps |
| `Plot_Feature_Importance.png` · `Plot_ObsVsPred.png` · `Plot_Model_Comparison.png` | Diagnostics |
| `RF_All_Results.csv` | Every split × CV result, with the blend on the same rows |
| `Robust_Eval.csv` | Test R² over 50 repeated splits, Random Forest vs blend |
| `Spatial_Validation.csv` · `Spatial_Validation_Distances.csv` | Random vs spatially blocked CV and the distance check |
| `SOC_Map_Summary.csv` · `feature_importance.csv` | Map statistics and covariate ranking |
| `SOC_FinalRF_Model.rds` · `SOC_FinalBlend_Model.rds` | Saved models (not committed) |

Figures and CSVs from the latest run are committed in [`output/`](output/).

</details>

## ⚠️ Limitations

- Accuracy is low: the covariates explain roughly 5 % (Random Forest) to 13 % (blend) of SOC variation at new points inside the watershed, and nothing outside it.
- 80 samples over ~117 km² is sparse. More samples, or rasters for further covariates (NDMI, BSI, NDVI statistics, CEC), would help more than another algorithm.
- Sample points and covariate rasters are not in the repository, so the map cannot be regenerated from a fresh clone.

## 👥 Team

Semester 2 project, Indian Institute of Remote Sensing (IIRS-ISRO), Dehradun · M.Sc. Agriculture Analytics (DAU · AAU · IIRS-ISRO)

| Member | Links |
|---|---|
| **Sathwik Ramaka** | [LinkedIn](https://www.linkedin.com/in/sathwikramaka/) · [GitHub](https://github.com/sathwikramaka) |
| **Kadiri Yaswanthi** | [LinkedIn](https://www.linkedin.com/in/kadiri-yaswanthi-9a632b204/) · [GitHub](https://github.com/YASWANTHI2410) |

<sub>MIT licence</sub>
