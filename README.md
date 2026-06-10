# Soil Organic Carbon Mapping — Dhanolti, Uttarakhand

## Overview
Digital soil mapping of Soil Organic Carbon (SOC) for the Dhanolti 
region of Uttarakhand using Random Forest machine learning with 
Google Earth Engine-derived environmental covariates.

## Key Results
- Test R² = 0.44
- Covariates: NDVI, EVI, terrain derivatives, spectral bands (Sentinel-2)
- Model: Random Forest (Scikit-Learn)

## Tools & Libraries
- Google Earth Engine (Python API)
- Scikit-Learn
- GeoPandas, NumPy, Pandas
- QGIS (visualization)
- Matplotlib

## Methodology
1. Study area delineation and sampling design
2. GEE covariate extraction (spectral indices, terrain, climate)
3. Random Forest model training and cross-validation
4. SOC prediction map generation
5. Accuracy assessment (R², RMSE, MAE)

## Output Maps
[Insert map image here]

## How to Run
```bash
pip install -r requirements.txt
python soil_carbon_mapping.py
```

## Project Context
Completed as part of M.Sc. Agriculture Analytics at 
IIRS-ISRO, Dehradun (Semester 2, 2026).
