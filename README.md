# Soil Organic Carbon Mapping — Dhanolti, Uttarakhand

## Overview
Digital soil mapping of Soil Organic Carbon (SOC) for the Dhanolti region of Uttarakhand using the SCORPAN-E framework with Random Forest machine learning. Environmental covariates were derived from Landsat 8/9, ERA5 climate data, and SRTM terrain analysis via Google Earth Engine.

This project was completed as part of M.Sc. Agriculture Analytics at the Indian Institute of Remote Sensing (IIRS-ISRO), Dehradun.

---

## Key Results
- **Best Model:** Random Forest (test R² = 0.44)
- **Total covariates used:** 27 environmental variables
- **Framework:** SCORPAN-E (Soil, Climate, Organisms, Relief, Parent material, Age, Nearby soil, human activity)
- **Validation:** Train/test split with held-out test set

---

## Data Sources
| Source | Variables |
|--------|-----------|
| Landsat 8/9 (via GEE) | Spectral bands, NDVI, EVI, SAVI |
| SRTM (via GEE) | Elevation, Slope, Aspect, Curvature |
| ERA5 (via GEE) | Mean temperature, precipitation |
| MODIS (via GEE) | Land cover, vegetation indices |
| Field sampling | Soil organic carbon (ground truth) |

---

## Tools & Libraries
- **Language:** R / RStudio
- **Machine Learning:** `randomForest` package
- **Spatial Analysis:** `terra`, `raster`, `sf`
- **Covariate Extraction:** Google Earth Engine (Python API)
- **Visualization:** QGIS, `ggplot2`

---

## Methodology
1. **Study area delineation** — Dhanolti region, Uttarakhand, India
2. **Covariate extraction** — 27 variables extracted via Google Earth Engine
3. **Soil sampling** — Field SOC measurements used as training data
4. **Feature selection** — Variable importance ranking via Random Forest
5. **Model training** — Random Forest with cross-validation
6. **Accuracy assessment** — R², RMSE, MAE on held-out test set
7. **SOC prediction map** — Spatial prediction across study region

---

## Repository Structure
