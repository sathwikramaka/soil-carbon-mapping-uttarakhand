# =============================================================================
# SOC_MASTER.R — Digital Soil Organic Carbon Mapping
# Study Area: Dhanolti, Uttarakhand
# Model      : Random Forest | Splits: 70:30 & 80:20 | CV: 5-fold, 10-fold, LOOCV
# =============================================================================

# ── 0. PACKAGES ───────────────────────────────────────────────────────────────
pkgs <- c("randomForest","caret","terra","sf","ggplot2","dplyr",
          "tidyr","tibble","gridExtra","viridis","RColorBrewer","readr")
for (p in pkgs) {
  if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
  library(p, character.only=TRUE)
}
set.seed(42)

# =============================================================================
# !! UPDATE THESE TWO PATHS ONLY !!
# =============================================================================
PROJECT_DIR <- "C:/Case_Study_SOC/SOC_Project"
CSV_NAME    <- "SOC_COMPLETE_FINAL_3.csv"   # CSV must be in PROJECT_DIR/data/
# =============================================================================

setwd(PROJECT_DIR)
if (!dir.exists("output")) dir.create("output")

DATA_DIR    <- file.path(PROJECT_DIR, "data")
RASTER_DIR  <- file.path(DATA_DIR, "rasters")
BOUNDARY_DIR<- file.path(DATA_DIR, "boundary")
CSV_PATH    <- file.path(DATA_DIR, CSV_NAME)

cat(strrep("=",65), "\n")
cat("  SOC DIGITAL SOIL MAPPING — DHANOLTI, UTTARAKHAND\n")
cat(strrep("=",65), "\n\n")

# =============================================================================
# SECTION 1 — LOAD & CLEAN DATA
# =============================================================================
cat("[ 1/6 ] LOADING & CLEANING DATA\n", strrep("-",45), "\n")

df_raw <- read_csv(CSV_PATH, show_col_types=FALSE)
cat("Raw data:", nrow(df_raw), "rows |", ncol(df_raw), "cols\n")
cat("Columns:", paste(names(df_raw), collapse=", "), "\n\n")

# ── Drop coordinate / ID columns ─────────────────────────────
drop_pat <- "^sno$|^id$|^FID$|^lat$|^Lat$|^long$|^Long$|^lon$|
             ^x$|^X$|^y$|^Y$|^longitude$|^latitude$|^OBJECTID$|geometry"
drop_cols <- grep(drop_pat, names(df_raw), ignore.case=TRUE, value=TRUE)
df <- df_raw %>% select(-any_of(drop_cols))
cat("Dropped:", paste(drop_cols, collapse=", "), "\n")

# ── Identify SOC column ───────────────────────────────────────
soc_col <- grep("^soc$|^SOC$|^soc_pct$|^OC$", names(df),
                ignore.case=TRUE, value=TRUE)[1]
if (is.na(soc_col)) stop("SOC column not found. Check CSV.")
names(df)[names(df)==soc_col] <- "soc"
cat("SOC column:", soc_col, "\n")

# ── Convert to numeric ────────────────────────────────────────
df <- df %>% mutate(across(-soc, ~suppressWarnings(as.numeric(.x))))

# ── SOC range filter ──────────────────────────────────────────
df <- df %>% filter(!is.na(soc), soc > 0, soc < 25)

# ── Remove near-zero variance columns ────────────────────────
pred_cols <- setdiff(names(df), "soc")
nzv <- sapply(df[,pred_cols], function(x)
  length(unique(na.omit(x))) <= 2 | sd(na.omit(x)) < 1e-8)
if (any(nzv)) {
  cat("Removing NZV:", paste(names(nzv)[nzv], collapse=", "), "\n")
  df <- df %>% select(-any_of(names(nzv)[nzv]))
}
pred_cols <- setdiff(names(df), "soc")

# ── Outlier removal: 3×IQR on SOC only (preserve predictor range) ──
q  <- quantile(df$soc, c(0.25,0.75))
iq <- q[2]-q[1]
df <- df %>% filter(soc >= q[1]-3*iq, soc <= q[2]+3*iq)

# ── Median imputation ─────────────────────────────────────────
for (col in pred_cols)
  if (any(is.na(df[[col]])))
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm=TRUE)

cat("Clean dataset:", nrow(df), "rows |", length(pred_cols), "predictors\n")
cat("SOC: mean=", round(mean(df$soc),2),
    "| sd=", round(sd(df$soc),2),
    "| min=", round(min(df$soc),2),
    "| max=", round(max(df$soc),2), "\n\n")

# ── Define predictor set (matching raster files available) ────
# These are the 27 predictors you have TIF files for:
raster_features <- c(
  "slope_1","twi_1","elevation_1","aspect_1",
  "p_curvature_1","pl_curvature_1",
  "mat_celsius","map_mm","precip_seasonality",
  "NDVI","EVI","NDWI","SAVI",
  "ndvi_slope","ndvi_intercept","ndvi_max",
  "monsoon_ndvi_slope","winter_ndvi_slope",
  "dist_to_stream","lithology","LULC",
  "pH_water","sand_pct","clay_pct","silt_pct",
  "nitrogen_gkg","BD_kgdm3"
)
# Keep only those present in CSV
raster_features <- raster_features[raster_features %in% names(df)]
cat("Predictors with raster files:", length(raster_features), "\n")
cat(paste(raster_features, collapse=", "), "\n\n")

model_df <- df %>% select(soc, all_of(raster_features)) %>% na.omit()
cat("Final modeling dataset:", nrow(model_df), "samples,",
    ncol(model_df)-1, "predictors\n\n")

write.csv(model_df, "output/data_clean.csv", row.names=FALSE)

# =============================================================================
# SECTION 2 — HELPER FUNCTIONS
# =============================================================================
calc_metrics <- function(obs, pred) {
  rmse <- sqrt(mean((obs-pred)^2))
  mae  <- mean(abs(obs-pred))
  r2   <- 1 - sum((obs-pred)^2)/sum((obs-mean(obs))^2)
  bias <- mean(pred-obs)
  rpiq <- (quantile(obs,0.75)-quantile(obs,0.25))/rmse
  tibble(RMSE=round(rmse,4), MAE=round(mae,4),
         R2=round(r2,4), Bias=round(bias,4), RPIQ=round(rpiq,4))
}

# =============================================================================
# SECTION 3 — FEATURE SELECTION (RF importance on full dataset)
# =============================================================================
cat("[ 2/6 ] FEATURE SELECTION\n", strrep("-",45), "\n")

# Train preliminary RF on full data to rank features
set.seed(42)
rf_prelim <- randomForest(
  x=model_df[, raster_features],
  y=model_df$soc,
  ntree=500,
  mtry=floor(sqrt(length(raster_features))),
  importance=TRUE
)

imp_df <- data.frame(
  Variable  = rownames(rf_prelim$importance),
  IncMSE    = rf_prelim$importance[,"%IncMSE"],
  NodePurity= rf_prelim$importance[,"IncNodePurity"]
) %>% arrange(desc(IncMSE))

cat("Feature Importance Ranking:\n")
print(imp_df, row.names=FALSE)
write.csv(imp_df, "output/feature_importance.csv", row.names=FALSE)

# Keep features with IncMSE > 0 (removes truly harmful predictors)
sel_features <- imp_df %>% filter(IncMSE > 0) %>% pull(Variable)
cat("\nSelected", length(sel_features), "features (IncMSE > 0)\n\n")

# Update model_df to selected features
model_df <- model_df %>% select(soc, all_of(sel_features))

# ── Feature importance plot ───────────────────────────────────
p_imp <- ggplot(imp_df %>% filter(IncMSE > 0),
                aes(x=reorder(Variable, IncMSE), y=IncMSE, fill=IncMSE)) +
  geom_col(width=0.7) +
  scale_fill_viridis_c(option="plasma") +
  coord_flip() +
  labs(title="RF Variable Importance — %IncMSE",
       subtitle=paste0("SOC Prediction | Dhanolti | ",
                       length(sel_features), " selected features"),
       x=NULL, y="%IncMSE") +
  theme_bw(base_size=12) +
  theme(plot.title=element_text(face="bold"), legend.position="none")
ggsave("output/Plot_Feature_Importance.png", p_imp,
       width=10, height=7, dpi=200, bg="white")
cat("Saved: output/Plot_Feature_Importance.png\n\n")

# =============================================================================
# SECTION 4 — TRAIN/TEST SPLITS + RF + CV
# =============================================================================
cat("[ 3/6 ] MODEL TRAINING — RF with 70:30 & 80:20 | 5-fold, 10-fold, LOOCV\n")
cat(strrep("-",45), "\n")

# mtry tuning values
n_pred    <- length(sel_features)
mtry_vals <- unique(c(
  max(1, floor(sqrt(n_pred)/2)),
  max(1, floor(sqrt(n_pred))),
  max(1, floor(sqrt(n_pred)*1.5)),
  max(1, floor(n_pred/3))
))

splits_cfg <- list(
  "70:30" = 0.70,
  "80:20" = 0.80
)

cv_cfg <- list(
  "5-fold"  = trainControl(method="cv",    number=5,  savePredictions="final"),
  "10-fold" = trainControl(method="cv",    number=10, savePredictions="final"),
  "LOOCV"   = trainControl(method="LOOCV",            savePredictions="final")
)

all_results <- list()
all_models  <- list()
holdout_results <- list()

for (sp_name in names(splits_cfg)) {
  frac <- splits_cfg[[sp_name]]

  # Stratified split via createDataPartition
  set.seed(42)
  train_idx  <- createDataPartition(model_df$soc, p=frac, list=FALSE)
  train_data <- model_df[ train_idx, ]
  test_data  <- model_df[-train_idx, ]

  cat(sprintf("\n%s | Train: %d | Test: %d\n",
              sp_name, nrow(train_data), nrow(test_data)))

  # ── Baseline RF for holdout metrics ──────────────────────
  set.seed(42)
  rf_base <- randomForest(
    x=train_data[,sel_features], y=train_data$soc,
    ntree=1000, mtry=floor(sqrt(n_pred)), importance=TRUE
  )
  pred_test  <- predict(rf_base, test_data[,sel_features])
  pred_train <- predict(rf_base, train_data[,sel_features])

  m_test  <- calc_metrics(test_data$soc,  pred_test)
  m_train <- calc_metrics(train_data$soc, pred_train)

  cat(sprintf("  Holdout Test  R2=%.3f RMSE=%.3f\n",
              m_test$R2, m_test$RMSE))
  cat(sprintf("  Holdout Train R2=%.3f RMSE=%.3f\n",
              m_train$R2, m_train$RMSE))

  holdout_results[[sp_name]] <- list(
    model=rf_base, train=train_data, test=test_data,
    obs_test=test_data$soc, pred_test=pred_test,
    obs_train=train_data$soc, pred_train=pred_train,
    m_test=m_test, m_train=m_train
  )

  # ── CV with mtry tuning ───────────────────────────────────
  for (cv_name in names(cv_cfg)) {
    cat(sprintf("  CV: %-10s ... ", cv_name))

    set.seed(42)
    cv_fit <- tryCatch(
      train(soc ~ ., data=train_data,
            method="rf",
            trControl=cv_cfg[[cv_name]],
            tuneGrid=data.frame(mtry=mtry_vals),
            ntree=1000,
            importance=TRUE),
      error=function(e){ cat("WARN:", e$message, "\n"); NULL }
    )
    if (is.null(cv_fit)) next

    best_mtry  <- cv_fit$bestTune$mtry
    cv_r2      <- max(cv_fit$results$Rsquared, na.rm=TRUE)
    cv_rmse    <- cv_fit$results$RMSE[which.max(cv_fit$results$Rsquared)]
    pred_te    <- predict(cv_fit, test_data)
    te_r2      <- calc_metrics(test_data$soc, pred_te)$R2
    te_rmse    <- calc_metrics(test_data$soc, pred_te)$RMSE

    row <- tibble(
      Split=sp_name, CV=cv_name, Best_mtry=best_mtry,
      Train_R2  = round(m_train$R2,   3),
      Train_RMSE= round(m_train$RMSE, 3),
      Test_R2   = round(te_r2,        3),
      Test_RMSE = round(te_rmse,      3),
      CV_R2     = round(cv_r2,        3),
      CV_RMSE   = round(cv_rmse,      3)
    )
    all_results[[length(all_results)+1]] <- row
    all_models[[paste0(sp_name,"_",cv_name)]] <- list(
      model=cv_fit, train=train_data, test=test_data,
      te_r2=te_r2, cv_r2=cv_r2
    )
    cat(sprintf("mtry=%d | Test R2=%.3f | CV R2=%.3f\n",
                best_mtry, te_r2, cv_r2))
  }
}

# ── Results table ─────────────────────────────────────────────
results_df <- bind_rows(all_results)
write.csv(results_df, "output/RF_All_Results.csv", row.names=FALSE)

cat("\n=== ALL MODEL RESULTS ===\n")
print(results_df %>% arrange(desc(Test_R2)))

# =============================================================================
# SECTION 5 — SELECT BEST MODEL
# =============================================================================
cat("\n[ 4/6 ] SELECTING BEST MODEL\n", strrep("-",45), "\n")

best_row <- results_df %>%
  arrange(desc(Test_R2), desc(CV_R2)) %>% slice(1)

best_key   <- paste0(best_row$Split, "_", best_row$CV)
best_obj   <- all_models[[best_key]]

cat(sprintf("BEST: Split=%s | CV=%s | Test R2=%.3f | CV R2=%.3f\n\n",
            best_row$Split, best_row$CV,
            best_row$Test_R2, best_row$CV_R2))

# ── CRITICAL: Retrain FINAL model on ALL data for mapping ────
# (Reference code approach — maximises information for spatial prediction)
cat("Retraining final RF on ALL", nrow(model_df), "samples...\n")
set.seed(42)
final_rf <- randomForest(
  soc ~ .,
  data       = model_df,
  ntree      = 1000,
  mtry       = best_row$Best_mtry,
  importance = TRUE,
  keep.forest= TRUE
)
oob_r2 <- round(1 - final_rf$mse[1000]/var(model_df$soc), 3)
cat(sprintf("Final RF OOB R2=%.3f | OOB RMSE=%.3f\n\n",
            oob_r2, round(sqrt(final_rf$mse[1000]),4)))

saveRDS(final_rf, "output/SOC_FinalRF_Model.rds")

# ── Observed vs Predicted plots ───────────────────────────────
make_scatter <- function(obs, pred, title_txt) {
  d    <- tibble(obs=obs, pred=pred)
  r2v  <- round(1-sum((obs-pred)^2)/sum((obs-mean(obs))^2), 3)
  rmse_v <- round(sqrt(mean((obs-pred)^2)), 3)
  ggplot(d, aes(obs,pred)) +
    geom_point(alpha=0.7, color="#2166AC", size=2.5) +
    geom_abline(slope=1, intercept=0, color="red", linetype="dashed", linewidth=0.8) +
    geom_smooth(method="lm", se=TRUE, color="#4DAC26",
                fill="#B8E186", alpha=0.3) +
    annotate("text", x=min(obs), y=max(pred),
             label=sprintf("R\u00b2 = %.3f\nRMSE = %.3f", r2v, rmse_v),
             hjust=0, vjust=1, size=4, fontface="italic") +
    labs(title=title_txt, x="Observed SOC (%)", y="Predicted SOC (%)") +
    theme_bw(base_size=12) +
    theme(plot.title=element_text(face="bold", size=11))
}

ho70 <- holdout_results[["70:30"]]
ho80 <- holdout_results[["80:20"]]

p1 <- make_scatter(ho80$obs_test,  ho80$pred_test,  "80:20 — Test Set")
p2 <- make_scatter(ho70$obs_test,  ho70$pred_test,  "70:30 — Test Set")
p3 <- make_scatter(ho80$obs_train, ho80$pred_train, "80:20 — Training Set")
p4 <- make_scatter(ho70$obs_train, ho70$pred_train, "70:30 — Training Set")

png("output/Plot_ObsVsPred.png", width=2400, height=2000, res=200)
grid.arrange(p1, p2, p3, p4, ncol=2,
             top="SOC Random Forest — Observed vs Predicted")
dev.off()
cat("Saved: output/Plot_ObsVsPred.png\n")

# ── Model comparison plot ─────────────────────────────────────
p_comp <- ggplot(results_df, aes(x=CV, y=Test_R2, fill=Split)) +
  geom_col(position=position_dodge(0.75), width=0.65) +
  geom_text(aes(label=sprintf("%.3f", Test_R2)),
            position=position_dodge(0.75), vjust=-0.4, size=3) +
  scale_fill_brewer(palette="Set2") +
  labs(title="RF Test R\u00b2 — Splits & CV Methods",
       x=NULL, y="Test R\u00b2", fill="Split") +
  theme_bw(base_size=12) +
  theme(plot.title=element_text(face="bold"))
ggsave("output/Plot_Model_Comparison.png", p_comp,
       width=10, height=6, dpi=200, bg="white")
cat("Saved: output/Plot_Model_Comparison.png\n\n")

# =============================================================================
# SECTION 6 — SPATIAL PREDICTION & MAPPING
# =============================================================================
cat("[ 5/6 ] SPATIAL PREDICTION & MAPPING\n", strrep("-",45), "\n")

# ── Raster filename → predictor name map ─────────────────────
# Exact filenames as in your data/rasters/ folder
raster_name_map <- c(
  "slope_1"            = "slope_1.tif",
  "twi_1"              = "twi_1.tif",
  "elevation_1"        = "elevation_1.tif",
  "aspect_1"           = "aspect_1.tif",
  "p_curvature_1"      = "p_curvature_1.tif",
  "pl_curvature_1"     = "pl_curvature_1.tif",
  "mat_celsius"        = "mat_celsius.tif",
  "map_mm"             = "map_mm.tif",
  "precip_seasonality" = "precip_seasonality.tif",
  "NDVI"               = "NDVI.tif",
  "EVI"                = "evi.tif",        # lowercase in your folder
  "NDWI"               = "NDWI.tif",
  "SAVI"               = "savi.tif",       # lowercase in your folder
  "ndvi_slope"         = "ndvi_slope.tif",
  "ndvi_intercept"     = "ndvi_intercept.tif",
  "ndvi_max"           = "ndvi_max.tif",
  "monsoon_ndvi_slope" = "monsoon_ndvi_slope.tif",
  "winter_ndvi_slope"  = "winter_ndvi_slope.tif",
  "dist_to_stream"     = "dist_to_stream.tif",
  "lithology"          = "lithology.tif",
  "LULC"               = "LULC.tif",
  "pH_water"           = "pH_water.tif",
  "sand_pct"           = "sand_pct.tif",
  "clay_pct"           = "clay_pct.tif",
  "silt_pct"           = "silt_pct.tif",
  "nitrogen_gkg"       = "nitrogen_gkg.tif",
  "BD_kgdm3"           = "BD_kgdm3.tif"
)

# ── Reference raster (slope_1 for CRS/extent) ────────────────
ref_path <- file.path(RASTER_DIR, "slope_1.tif")
if (!file.exists(ref_path))
  stop("slope_1.tif not found in data/rasters/")
ref <- rast(ref_path)
cat("Reference CRS:", crs(ref, describe=TRUE)$name, "\n")
cat("Reference res:", res(ref)[1], "m\n\n")

# ── Load & align rasters ──────────────────────────────────────
cat("Loading rasters...\n")
raster_list    <- list()
missing_rasters<- c()

for (pred_name in names(raster_name_map)) {
  if (!(pred_name %in% sel_features)) next   # skip if not selected
  fpath <- file.path(RASTER_DIR, raster_name_map[pred_name])
  if (file.exists(fpath)) {
    r <- rast(fpath)
    # Reproject if CRS differs
    if (nchar(crs(r)) > 0 && !same.crs(r, ref))
      r <- project(r, ref,
                   method=if (pred_name %in% c("lithology","LULC")) "near"
                           else "bilinear")
    # Crop then resample
    r <- crop(r, ext(ref))
    if (!compareGeom(r, ref, stopOnError=FALSE, res=TRUE))
      r <- resample(r, ref,
                    method=if (pred_name %in% c("lithology","LULC")) "near"
                            else "bilinear")
    names(r) <- pred_name
    nv <- sum(!is.na(values(r)))
    cat(sprintf("  %-22s : %d valid px\n", pred_name, nv))
    if (nv > 0) raster_list[[pred_name]] <- r
    else missing_rasters <- c(missing_rasters, pred_name)
  } else {
    cat(sprintf("  %-22s : NOT FOUND\n", pred_name))
    missing_rasters <- c(missing_rasters, pred_name)
  }
}

if (length(missing_rasters) > 0) {
  cat("\nMissing/empty rasters:", paste(missing_rasters, collapse=", "), "\n")
  cat("Retraining final RF without these variables...\n")
  map_features <- setdiff(sel_features, missing_rasters)
  map_df <- model_df %>% select(soc, all_of(map_features))
  set.seed(42)
  final_rf <- randomForest(soc ~ ., data=map_df, ntree=1000,
    mtry=max(1,floor(sqrt(length(map_features)))),
    importance=TRUE, keep.forest=TRUE)
  cat("Retrained OOB R2:",
      round(1-final_rf$mse[1000]/var(map_df$soc),3), "\n")
} else {
  map_features <- sel_features
}

# ── Build raster stack ────────────────────────────────────────
rast_stack        <- rast(raster_list[map_features])
names(rast_stack) <- map_features
cat(sprintf("\nStack: %d layers | %d x %d px\n",
            nlyr(rast_stack), nrow(rast_stack), ncol(rast_stack)))

# ── Per-layer diagnostic ──────────────────────────────────────
any_zero <- FALSE
for (nm in names(rast_stack)) {
  nv <- sum(!is.na(values(rast_stack[[nm]])))
  if (nv==0) { cat("!! EMPTY:", nm, "\n"); any_zero <- TRUE }
}
if (any_zero) stop("Empty layers found — fix rasters before predicting.")

# ── Predict SOC across landscape ─────────────────────────────
cat("\nPredicting SOC (1-5 min)...\n")
soc_map        <- terra::predict(rast_stack, final_rf, na.rm=TRUE)
names(soc_map) <- "SOC_pct"
n_valid        <- sum(!is.na(values(soc_map)))
cat("Valid predicted pixels:", n_valid, "\n")
if (n_valid==0) stop("All pixels NA after prediction.")

# ── Boundary mask ─────────────────────────────────────────────
bnd_files <- list.files(BOUNDARY_DIR, pattern="\\.shp$", full.names=TRUE)
if (length(bnd_files)==0) stop("No .shp in data/boundary/")
boundary      <- vect(bnd_files[1])
boundary      <- aggregate(boundary)
crs(boundary) <- crs(soc_map)   # assign UTM 44N (no .prj)

soc_masked <- mask(crop(soc_map, boundary), boundary)
n_masked   <- sum(!is.na(values(soc_masked)))
cat("Valid pixels after mask:", n_masked, "\n")
if (n_masked==0) stop("Boundary mask gave 0 pixels.")

# ── Save TIFs ────────────────────────────────────────────────
writeRaster(soc_masked, "output/SOC_Predicted.tif",
            overwrite=TRUE, datatype="FLT4S")
cat("Saved: output/SOC_Predicted.tif\n")

# ── Uncertainty map (SD across trees) ────────────────────────
cat("Computing uncertainty map (SD across trees)...\n")
rast_df      <- as.data.frame(rast_stack, na.rm=FALSE)
valid_mask   <- complete.cases(rast_df)
sd_vals      <- rep(NA_real_, nrow(rast_df))
if (sum(valid_mask) > 0) {
  tree_preds <- predict(final_rf, rast_df[valid_mask,],
                        predict.all=TRUE)$individual
  sd_vals[valid_mask] <- apply(tree_preds, 1, sd)
}
soc_sd        <- rast_stack[[1]]
values(soc_sd)<- sd_vals
names(soc_sd) <- "SOC_SD"
soc_sd_masked <- mask(crop(soc_sd, boundary), boundary)
writeRaster(soc_sd_masked, "output/SOC_Uncertainty.tif",
            overwrite=TRUE, datatype="FLT4S")
cat("Saved: output/SOC_Uncertainty.tif\n\n")

# =============================================================================
# SECTION 7 — MAP VISUALISATION (no pixelation, high quality)
# =============================================================================
cat("[ 6/6 ] MAP VISUALISATION\n", strrep("-",45), "\n")

# ── Area & stats ──────────────────────────────────────────────
soc_vals <- values(soc_masked, na.rm=TRUE)
res_m    <- res(soc_masked)
area_ha  <- if (is.lonlat(soc_masked)) {
  sum(values(cellSize(soc_masked, unit="m"))[!is.na(values(soc_masked))],
      na.rm=TRUE) / 10000
} else { n_masked * res_m[1] * res_m[2] / 10000 }

cat("\n")
cat(strrep("\u2550",50), "\n")
cat(sprintf("  Area       : %.0f ha\n",    area_ha))
cat(sprintf("  Min SOC    : %.3f %%\n",    min(soc_vals)))
cat(sprintf("  Max SOC    : %.3f %%\n",    max(soc_vals)))
cat(sprintf("  Mean SOC   : %.3f %%\n",    mean(soc_vals)))
cat(sprintf("  Median SOC : %.3f %%\n",    median(soc_vals)))
cat(sprintf("  Std Dev    : %.3f %%\n",    sd(soc_vals)))
cat(sprintf("  OOB R\u00b2     : %.3f\n",  oob_r2))
cat(sprintf("  Best Split : %s | CV: %s\n",best_row$Split, best_row$CV))
cat(sprintf("  Test R\u00b2   : %.3f\n",   best_row$Test_R2))
cat(strrep("\u2550",50), "\n\n")

# Convert to data frame for ggplot
# Use coord_equal() — prevents pixelation (key fix)
soc_df <- as.data.frame(soc_masked, xy=TRUE) %>%
  filter(!is.na(SOC_pct))

# Convert boundary to coordinates for geom_path (compatible with coord_equal)
bnd_sf     <- st_as_sf(boundary)
bnd_coords <- as.data.frame(
  st_coordinates(st_cast(bnd_sf, "MULTILINESTRING"))
)
# L2 groups separate polygon rings — use as group
bnd_coords$group <- paste(bnd_coords$L1, bnd_coords$L2, sep="_")

subtitle_txt <- sprintf(
  "RF (ntree=1000) | Mean=%.2f%% | Area=%.0f ha | n=%d | Best: %s/%s | Test R\u00b2=%.3f",
  mean(soc_vals), area_ha, nrow(model_df),
  best_row$Split, best_row$CV, best_row$Test_R2
)

# ── Continuous SOC map (viridis — no pixelation) ──────────────
p_soc <- ggplot() +
  geom_raster(data=soc_df, aes(x=x, y=y, fill=SOC_pct),
              interpolate=TRUE) +            # interpolate=TRUE smooths map
  scale_fill_viridis_c(
    option   = "D",
    name     = "SOC (%)",
    na.value = "transparent"
  ) +
  geom_path(data=bnd_coords, aes(x=X, y=Y, group=group),
            color="black", linewidth=1.0, inherit.aes=FALSE) +
  coord_equal() +
  labs(
    title    = "Predicted Soil Organic Carbon \u2014 Dhanolti, Uttarakhand",
    subtitle = subtitle_txt,
    x        = "Easting (m)", y = "Northing (m)",
    caption  = "SCORPAN-E Framework | Random Forest | 30m resolution"
  ) +
  theme_bw(base_size=12) +
  theme(
    plot.title    = element_text(face="bold", size=13),
    plot.subtitle = element_text(size=8.5, color="gray40"),
    legend.key.height = unit(1.5,"cm")
  )
ggsave("output/SOC_Map.png", p_soc,
       width=11, height=9, dpi=250, bg="white")
cat("Saved: output/SOC_Map.png\n")

# ── Zone classification map ───────────────────────────────────
soc_zones <- classify(soc_masked,
  rcl=matrix(c(-Inf,1,1, 1,2,2, 2,3,3, 3,4,4, 4,Inf,5),
             ncol=3, byrow=TRUE))
zone_labels <- c("Very Low <1%","Low 1-2%","Medium 2-3%",
                 "High 3-4%","Very High >4%")
zone_colors <- c("#8B1A1A","#D2691E","#F4A460","#90EE90","#228B22")

writeRaster(soc_zones, "output/SOC_Zones.tif",
            overwrite=TRUE, datatype="INT1U")

zone_df <- as.data.frame(soc_zones, xy=TRUE) %>% filter(!is.na(SOC_pct))
names(zone_df)[3] <- "Zone"
zone_df$Class <- factor(zone_df$Zone, levels=1:5, labels=zone_labels)

p_zone <- ggplot() +
  geom_raster(data=zone_df, aes(x=x, y=y, fill=Class),
              interpolate=FALSE) +
  scale_fill_manual(values=setNames(zone_colors, zone_labels),
                    name="SOC Class", drop=FALSE) +
  geom_path(data=bnd_coords, aes(x=X, y=Y, group=group),
            color="black", linewidth=1.0, inherit.aes=FALSE) +
  coord_equal() +
  labs(title="SOC Zone Map \u2014 Dhanolti, Uttarakhand",
       x="Easting (m)", y="Northing (m)",
       caption="SCORPAN-E | Random Forest | 30m") +
  theme_bw(base_size=12) +
  theme(plot.title=element_text(face="bold", size=13))
ggsave("output/SOC_Zone_Map.png", p_zone,
       width=11, height=9, dpi=250, bg="white")
cat("Saved: output/SOC_Zone_Map.png\n")

# ── Uncertainty map ───────────────────────────────────────────
sd_df <- as.data.frame(soc_sd_masked, xy=TRUE) %>%
  filter(!is.na(SOC_SD))
p_sd <- ggplot() +
  geom_raster(data=sd_df, aes(x=x, y=y, fill=SOC_SD),
              interpolate=TRUE) +
  scale_fill_distiller(palette="YlOrRd", direction=1,
                       name="SD (%)", na.value="transparent") +
  geom_path(data=bnd_coords, aes(x=X, y=Y, group=group),
            color="black", linewidth=1.0, inherit.aes=FALSE) +
  coord_equal() +
  labs(title="SOC Prediction Uncertainty \u2014 Dhanolti",
       subtitle="Standard Deviation across 1000 RF Trees",
       x="Easting (m)", y="Northing (m)") +
  theme_bw(base_size=12) +
  theme(plot.title=element_text(face="bold", size=13))
ggsave("output/SOC_Uncertainty_Map.png", p_sd,
       width=11, height=9, dpi=250, bg="white")
cat("Saved: output/SOC_Uncertainty_Map.png\n")

# ── Summary CSV ───────────────────────────────────────────────
write.csv(tibble(
  Metric=c("Area (ha)","Min SOC (%)","Max SOC (%)","Mean SOC (%)",
           "Median SOC (%)","Std Dev (%)","OOB R2",
           "Best Split","Best CV","Test R2","Test RMSE",
           "Variables Used","N Samples"),
  Value=c(round(area_ha,1), round(min(soc_vals),3),
          round(max(soc_vals),3), round(mean(soc_vals),3),
          round(median(soc_vals),3), round(sd(soc_vals),3),
          oob_r2, best_row$Split, best_row$CV,
          best_row$Test_R2, best_row$Test_RMSE,
          paste(map_features, collapse="; "),
          nrow(model_df))
), "output/SOC_Map_Summary.csv", row.names=FALSE)

write.csv(results_df, "output/RF_All_Results.csv", row.names=FALSE)

# ── Final banner ──────────────────────────────────────────────
cat("\n", strrep("\u2550",55), "\n")
cat("  PIPELINE COMPLETE\n")
cat(strrep("\u2550",55), "\n")
cat("  output/SOC_Predicted.tif\n")
cat("  output/SOC_Uncertainty.tif\n")
cat("  output/SOC_Zones.tif\n")
cat("  output/SOC_Map.png\n")
cat("  output/SOC_Zone_Map.png\n")
cat("  output/SOC_Uncertainty_Map.png\n")
cat("  output/Plot_ObsVsPred.png\n")
cat("  output/Plot_Model_Comparison.png\n")
cat("  output/Plot_Feature_Importance.png\n")
cat("  output/RF_All_Results.csv\n")
cat("  output/SOC_Map_Summary.csv\n")
cat("  output/SOC_FinalRF_Model.rds\n")
cat(strrep("\u2550",55), "\n")
