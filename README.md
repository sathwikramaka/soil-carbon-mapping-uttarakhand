# Soil Organic Carbon Mapping — Dhanolti, Uttarakhand

Digital soil mapping of Soil Organic Carbon (SOC) for the Dhanolti region,
Uttarakhand is using the **SCORPAN-E framework** with **Random Forest** machine
learning. Environmental covariates were extracted from satellite and terrain
data via Google Earth Engine.

**Program:** M.Sc. Agriculture Analytics — IIRS-ISRO, Dehradun (Semester 2, 2026)  
**Authors:** Sathwik Ramaka · K. Yaswanthi

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Test R² | 0.44 |
| Splits tested | 70:30 and 80:20 |
| Cross-validation | 5-fold, 10-fold, LOOCV |
| Final model (ntree) | 1000 trees |
| Resolution | 30 m |
| Framework | SCORPAN-E |

---

## Covariates (27 Environmental Predictors)

| Category | Variables |
|----------|-----------|
| **Terrain** | Slope, TWI, Elevation, Aspect, Plan Curvature, Profile Curvature |
| **Climate** | Mean Annual Temperature, Mean Annual Precipitation, Precipitation Seasonality |
| **Vegetation** | NDVI, EVI, NDWI, SAVI, NDVI Slope, NDVI Intercept, NDVI Max, Monsoon NDVI Slope, Winter NDVI Slope |
| **Hydrology** | Distance to Stream |
| **Land** | Lithology, LULC |
| **Soil Properties** | pH, Sand %, Clay %, Silt %, Nitrogen (g/kg), Bulk Density (kg/dm³) |

---

## Pipeline Overview

The script (`SOC_MASTER.R`) runs a full 6-section pipeline:

Section 1 → Data loading, cleaning, outlier removal, median imputation

Section 2 → Feature selection via RF variable importance (%IncMSE)

Section 3 → Model training: RF across 70:30 & 80:20 splits + 5-fold/10-fold/LOOCV

Section 4 → Best model selection + final RF (ntree=1000) training

Section 5 → Spatial prediction → SOC map + uncertainty map (SD across trees)

Section 6 → Map visualisation + output export

---

## Tools & Libraries

- **Language:** R / RStudio
- **ML:** `randomForest`, `caret`
- **Spatial:** `terra`, `sf`
- **Covariate Source:** Google Earth Engine
- **Visualisation:** `ggplot2`, `viridis`, `gridExtra`
- **Data:** `dplyr`, `tidyr`, `readr`

---

## Outputs

| File | Description |
|------|-------------|
| `output/SOC_Predicted.tif` | SOC prediction raster (30m) |
| `output/SOC_Uncertainty.tif` | SD across 1000 RF trees |
| `output/SOC_Zones.tif` | Classified SOC zones (5 classes) |
| `output/SOC_Map.png` | Continuous SOC map |
| `output/SOC_Zone_Map.png` | Classified zone map |
| `output/SOC_Uncertainty_Map.png` | Uncertainty map |
| `output/Plot_Feature_Importance.png` | Variable importance chart |
| `output/Plot_ObsVsPred.png` | Observed vs Predicted plot |
| `output/Plot_Model_Comparison.png` | Cross-validation comparison |
| `output/RF_All_Results.csv` | All model results |
| `output/SOC_Map_Summary.csv` | Summary statistics |
| `output/SOC_FinalRF_Model.rds` | Saved final RF model |

---

## How to Run

**1. Set up your paths** — edit only these two lines at the top of `SOC_MASTER.R`:

```r
PROJECT_DIR <- "path/to/your/project/folder"
CSV_NAME    <- "your_soc_data.csv"   # must be inside PROJECT_DIR/data/
```

**2. Folder structure required:**

PROJECT_DIR/

├── data/

│   ├── your_soc_data.csv

│   ├── rasters/          ← 27 .tif covariate files

│   └── boundary/         ← .shp study area boundary

└── output/               ← auto-created by script

**3. Install packages and run:**

```r
source("SOC_MASTER.R")
```

All packages install automatically on first run.

---

## Repository Structure

soil-carbon-mapping-uttarakhand/

├── SOC_MASTER.R     # Complete 6-section modelling pipeline

├── README.md        # Project documentation

├── .gitignore       # R gitignore

└── LICENSE          # MIT License

---

## Author

**Sathwik Ramaka**  
M.Sc. Agriculture Analytics | Remote Sensing & Carbon MRV  
[LinkedIn](https://linkedin.com/[in/YOUR-LINKEDIN-URL](https://www.linkedin.com/in/sathwik-ramaka-1ba40227a?utm_source=share_via&utm_content=profile&utm_medium=member_android)) · [GitHub](https://github.com/sathwikramaka)
