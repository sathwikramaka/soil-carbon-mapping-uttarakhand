# =============================================================================
#  SOC_MASTER.R — Digital Soil Organic Carbon Mapping, Dhanolti (Uttarakhand)
# =============================================================================
#  1. Load and clean the field samples
#  2. Select covariates by Random Forest permutation importance (%IncMSE > 0)
#  3. Train Random Forest on 70:30 and 80:20 splits; tune mtry with 5-fold,
#     10-fold and leave-one-out cross-validation
#  4. Choose the best model; robustness check over repeated splits
#  5. Predict SOC and its uncertainty at 30 m; classify SOC zones
#  6. Spatial validation: random vs spatially blocked cross-validation
#
#  Mapping model (FINAL_MODEL)
#    "RF"     Random Forest on the selected covariates
#    "Blend"  mean of (a) a Random Forest on the covariates plus oblique
#             geographic coordinates and (b) a radial support-vector regression
#
#  Folders: data/ (CSV, rasters/, boundary/) beside this script or one level
#  up. Results are written to output/ beside this script.
# =============================================================================


# ── Settings ─────────────────────────────────────────────────────────────────
CSV_NAME       <- "SOC_COMPLETE_FINAL_3.csv"
FINAL_MODEL    <- "Blend"                       # "RF" or "Blend"
SEED           <- 42
N_TREE         <- 1000
RUN_ROBUST     <- TRUE                          # section 4b
RUN_SPATIAL_CV <- TRUE                          # section 6
N_REPEATS      <- 50                            # repeated splits in 4b
N_CV_REPEATS   <- 20                            # cross-validation repetitions in 6
OBLIQUE_ANGLES <- c(0, 30, 60, 90, 120, 150)    # degrees

SPLITS <- c("70:30" = 0.70, "80:20" = 0.80)

# Covariate -> raster file in data/rasters/ (order matters for reproducibility)
RASTER_FILES <- c(
  slope_1 = "slope_1.tif", twi_1 = "twi_1.tif", elevation_1 = "elevation_1.tif",
  aspect_1 = "aspect_1.tif", p_curvature_1 = "p_curvature_1.tif",
  pl_curvature_1 = "pl_curvature_1.tif", mat_celsius = "mat_celsius.tif",
  map_mm = "map_mm.tif", precip_seasonality = "precip_seasonality.tif",
  NDVI = "NDVI.tif", EVI = "evi.tif", NDWI = "NDWI.tif", SAVI = "savi.tif",
  ndvi_slope = "ndvi_slope.tif", ndvi_intercept = "ndvi_intercept.tif",
  ndvi_max = "ndvi_max.tif", monsoon_ndvi_slope = "monsoon_ndvi_slope.tif",
  winter_ndvi_slope = "winter_ndvi_slope.tif", dist_to_stream = "dist_to_stream.tif",
  lithology = "lithology.tif", LULC = "LULC.tif", pH_water = "pH_water.tif",
  sand_pct = "sand_pct.tif", clay_pct = "clay_pct.tif", silt_pct = "silt_pct.tif",
  nitrogen_gkg = "nitrogen_gkg.tif", BD_kgdm3 = "BD_kgdm3.tif"
)
CATEGORICAL <- c("lithology", "LULC")

ZONES <- data.frame(
  label = c("Very Low <1%", "Low 1-2%", "Medium 2-3%", "High 3-4%", "Very High >4%"),
  color = c("#8B1A1A", "#D2691E", "#F4A460", "#90EE90", "#228B22")
)
MODEL_LABEL <- c(RF = "Random Forest", Blend = "RF (oblique coords) + SVR")


# ── Packages ─────────────────────────────────────────────────────────────────
# On Windows, use the PROJ/GDAL data bundled with terra and sf rather than data
# exported by other GIS installs (e.g. PostGIS), which causes proj.db warnings.
if (.Platform$OS.type == "windows") Sys.unsetenv(c("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"))

pkgs <- c("randomForest", "caret", "e1071", "terra", "sf",
          "ggplot2", "gridExtra", "readr", "dplyr")
if (getOption("repos")["CRAN"] %in% c(NA, "@CRAN@")) {
  options(repos = "https://cloud.r-project.org")
}
missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) install.packages(missing_pkgs)
invisible(lapply(pkgs, library, character.only = TRUE))
library(parallel)


# ── Paths ────────────────────────────────────────────────────────────────────
script_dir <- function() {
  file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(file_arg)) return(dirname(normalizePath(file_arg)))   # Rscript
  for (i in rev(seq_len(sys.nframe()))) {                            # source()
    f <- sys.frame(i)$ofile
    if (!is.null(f)) return(dirname(normalizePath(f)))
  }
  getwd()
}

SCRIPT_DIR <- script_dir()
DATA_DIR   <- Find(dir.exists, file.path(c(SCRIPT_DIR, dirname(SCRIPT_DIR)), "data"))
if (is.null(DATA_DIR)) stop("data/ folder not found beside or above ", SCRIPT_DIR)
OUT_DIR <- file.path(SCRIPT_DIR, "output")
dir.create(OUT_DIR, showWarnings = FALSE)
out <- function(name) file.path(OUT_DIR, name)


# ── Helpers: metrics and logging ─────────────────────────────────────────────
log_step <- function(title) cat("\n", title, "\n", strrep("-", 60), "\n", sep = "")

r2 <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

metrics <- function(obs, pred) {
  rmse <- sqrt(mean((obs - pred)^2))
  round(c(R2   = r2(obs, pred),
          RMSE = rmse,
          MAE  = mean(abs(obs - pred)),
          Bias = mean(pred - obs),
          RPIQ = unname(diff(quantile(obs, c(0.25, 0.75)))) / rmse), 4)
}


# ── Helpers: models ──────────────────────────────────────────────────────────
# 500-tree Random Forest; %IncMSE per covariate (covariate order)
rank_covariates <- function(d, covariates) {
  set.seed(SEED)
  rf <- randomForest(x = d[, covariates], y = d$soc, ntree = 500,
                     mtry = floor(sqrt(length(covariates))), importance = TRUE)
  data.frame(Variable   = covariates,
             IncMSE     = rf$importance[, "%IncMSE"],
             NodePurity = rf$importance[, "IncNodePurity"],
             row.names  = NULL)
}

# Covariates kept by a ranking: %IncMSE > 0 (top 5 if none)
kept_covariates <- function(imp) {
  kept <- imp$Variable[imp$IncMSE > 0]
  if (length(kept)) kept else head(imp$Variable[order(-imp$IncMSE)], 5)
}

# Oblique geographic coordinates: easting/northing projected onto rotated axes,
# so the forest can use location without axis-aligned stripes in the map
oblique <- function(xy) {
  a <- OBLIQUE_ANGLES * pi / 180
  m <- outer(xy[, 1], cos(a)) + outer(xy[, 2], sin(a))
  colnames(m) <- paste0("obl_", OBLIQUE_ANGLES)
  m
}

# Radial SVR. e1071 standardises SOC internally, so cost and epsilon are divided
# by sd(SOC) to set them on the original scale (C = 1, epsilon = 0.2 %).
fit_svr <- function(X, y) {
  s <- sd(y)
  svm(x = as.matrix(X), y = y, type = "eps-regression", kernel = "radial",
      gamma = 1 / ncol(X), cost = 1 / s, epsilon = 0.2 / s)
}

fit_blend <- function(X, y, xy) {
  Xo <- cbind(as.data.frame(X), oblique(xy))
  list(rf = randomForest(x = Xo, y = y, ntree = N_TREE, mtry = floor(sqrt(ncol(Xo)))),
       sv = fit_svr(X, y))
}

predict_blend <- function(model, X, xy) {
  rf_pred <- predict(model$rf, cbind(as.data.frame(X), oblique(xy)))
  (rf_pred + predict(model$sv, as.matrix(X))) / 2
}

# Mean and SD across trees, in chunks (avoids a cells x trees matrix in memory)
rf_mean_sd <- function(rf, newdata, chunk = 20000) {
  res <- matrix(NA_real_, nrow(newdata), 2, dimnames = list(NULL, c("mean", "sd")))
  for (i in seq(1, nrow(newdata), by = chunk)) {
    rows <- i:min(i + chunk - 1, nrow(newdata))
    trees <- predict(rf, newdata[rows, , drop = FALSE], predict.all = TRUE)$individual
    res[rows, ] <- cbind(rowMeans(trees), apply(trees, 1, sd))
  }
  res
}

# parLapply with the packages and objects that sections 4b and 6 need
par_map <- function(X, FUN) {
  cl <- makeCluster(max(1, detectCores() - 1))
  on.exit(stopCluster(cl))
  clusterEvalQ(cl, suppressPackageStartupMessages({
    library(randomForest); library(e1071); library(caret)
  }))
  clusterExport(cl, c("full_df", "xy", "covariates", "SEED", "N_TREE", "OBLIQUE_ANGLES",
                      "r2", "rank_covariates", "kept_covariates", "oblique",
                      "fit_svr", "fit_blend", "predict_blend"), envir = globalenv())
  parLapply(cl, X, FUN)
}


# ── Helpers: rasters and plots ───────────────────────────────────────────────
load_covariate <- function(name, ref) {
  path <- file.path(DATA_DIR, "rasters", RASTER_FILES[[name]])
  if (!file.exists(path)) stop("Raster not found: ", path)
  method <- if (name %in% CATEGORICAL) "near" else "bilinear"
  r <- rast(path)
  if (nchar(crs(r)) > 0 && !same.crs(r, ref)) r <- project(r, ref, method = method)
  r <- crop(r, ext(ref))
  if (!compareGeom(r, ref, stopOnError = FALSE, res = TRUE)) {
    r <- resample(r, ref, method = method)
  }
  names(r) <- name
  r
}

fill_raster <- function(template, cells, values, name) {
  v <- rep(NA_real_, ncell(template))
  v[cells] <- values
  r <- setValues(template, v)
  names(r) <- name
  r
}

theme_soc <- theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 8.5, colour = "grey40"))

scatter_plot <- function(obs, pred, title) {
  label <- sprintf("R² = %.3f\nRMSE = %.3f", r2(obs, pred), sqrt(mean((obs - pred)^2)))
  ggplot(data.frame(obs, pred), aes(obs, pred)) +
    geom_point(alpha = 0.7, colour = "#2166AC", size = 2.5) +
    geom_abline(colour = "red", linetype = "dashed", linewidth = 0.8) +
    geom_smooth(method = "lm", formula = y ~ x, colour = "#4DAC26", fill = "#B8E186", alpha = 0.3) +
    annotate("text", x = min(obs), y = max(pred), label = label,
             hjust = 0, vjust = 1, size = 4, fontface = "italic") +
    labs(title = title, x = "Observed SOC (%)", y = "Predicted SOC (%)") +
    theme_soc
}

raster_df <- function(r) {
  d <- as.data.frame(r, xy = TRUE, na.rm = TRUE)
  names(d)[3] <- "value"
  d
}

map_plot <- function(d, fill_scale, title, subtitle = NULL, interpolate = TRUE) {
  ggplot() +
    geom_raster(data = d, aes(x, y, fill = value), interpolate = interpolate, show.legend = TRUE) +
    fill_scale +
    geom_path(data = boundary_xy, aes(X, Y, group = group), linewidth = 1) +
    coord_equal() +
    labs(title = title, subtitle = subtitle, x = "Easting (m)", y = "Northing (m)",
         caption = paste0("SCORPAN-E | ", MODEL_LABEL[[FINAL_MODEL]], " | 30 m")) +
    theme_soc +
    theme(legend.key.height = unit(1.5, "cm"))
}


# =============================================================================
# 1. LOAD AND CLEAN DATA
# =============================================================================
log_step("[1/6] Load and clean data")

raw <- read_csv(file.path(DATA_DIR, CSV_NAME), show_col_types = FALSE)
find_col <- function(pattern) grep(pattern, names(raw), ignore.case = TRUE, value = TRUE)[1]
soc_col <- find_col("^(soc|soc_pct|oc)$")
lon_col <- find_col("^(long|lon|longitude)$")
lat_col <- find_col("^(lat|latitude)$")
if (anyNA(c(soc_col, lon_col, lat_col))) stop("CSV needs SOC, longitude and latitude columns")
id_cols <- grep("^(sno|id|fid|x|y|objectid)$|geometry", names(raw), ignore.case = TRUE, value = TRUE)

df <- raw %>%
  select(-any_of(id_cols)) %>%
  rename(soc = all_of(soc_col), lon = all_of(lon_col), lat = all_of(lat_col)) %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(.x)))) %>%
  filter(!is.na(soc), soc > 0, soc < 25)

# Near-zero-variance predictors
predictors <- setdiff(names(df), c("soc", "lon", "lat"))
nzv <- predictors[vapply(df[predictors], function(x)
  length(unique(na.omit(x))) <= 2 || sd(x, na.rm = TRUE) < 1e-8, logical(1))]
if (length(nzv)) cat("Removed near-zero-variance:", paste(nzv, collapse = ", "), "\n")
predictors <- setdiff(predictors, nzv)

# SOC outliers (3 x IQR), then median imputation of predictors
q   <- quantile(df$soc, c(0.25, 0.75))
iqr <- unname(diff(q))
df <- df %>%
  select(-all_of(nzv)) %>%
  filter(between(soc, q[1] - 3 * iqr, q[2] + 3 * iqr)) %>%
  mutate(across(all_of(predictors), ~ replace(.x, is.na(.x), median(.x, na.rm = TRUE))))

# Modelling table: SOC + covariates that have rasters; sample coordinates kept aside
covariates <- intersect(names(RASTER_FILES), names(df))
model_df   <- df %>% select(soc, all_of(covariates), lon, lat) %>% na.omit()
ref        <- rast(file.path(DATA_DIR, "rasters", "slope_1.tif"))
xy         <- crds(project(vect(as.data.frame(model_df), geom = c("lon", "lat"), crs = "EPSG:4326"),
                           crs(ref)))
model_df   <- model_df %>% select(-lon, -lat)
write.csv(model_df, out("data_clean.csv"), row.names = FALSE)

cat(sprintf("Samples: %d | covariates with rasters: %d\n", nrow(model_df), length(covariates)))
cat(sprintf("SOC (%%): mean %.2f | sd %.2f | min %.2f | max %.2f\n",
            mean(model_df$soc), sd(model_df$soc), min(model_df$soc), max(model_df$soc)))


# =============================================================================
# 2. COVARIATE SELECTION
# =============================================================================
log_step("[2/6] Covariate selection (Random Forest %IncMSE)")

importance_df <- rank_covariates(model_df, covariates)
importance_df <- importance_df[order(-importance_df$IncMSE), ]
selected      <- importance_df$Variable[importance_df$IncMSE > 0]
write.csv(importance_df, out("feature_importance.csv"), row.names = FALSE)
print(importance_df, row.names = FALSE, digits = 3)
cat(sprintf("\nSelected %d of %d covariates (%%IncMSE > 0)\n", length(selected), length(covariates)))

full_df  <- model_df                                  # all covariates (sections 4b, 6)
model_df <- model_df %>% select(soc, all_of(selected))

p <- ggplot(importance_df[importance_df$IncMSE > 0, ],
            aes(reorder(Variable, IncMSE), IncMSE, fill = IncMSE)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  scale_fill_viridis_c(option = "plasma") +
  coord_flip() +
  labs(title = "Random Forest variable importance (%IncMSE)",
       subtitle = sprintf("SOC, Dhanolti | %d selected covariates", length(selected)),
       x = NULL, y = "%IncMSE") +
  theme_soc
ggsave(out("Plot_Feature_Importance.png"), p, width = 10, height = 7, dpi = 200, bg = "white")


# =============================================================================
# 3. MODEL TRAINING — splits x cross-validation
# =============================================================================
log_step("[3/6] Model training: 70:30 and 80:20 splits; 5-fold, 10-fold, LOOCV")

n_sel     <- length(selected)
mtry_grid <- unique(pmax(1, floor(c(sqrt(n_sel) / 2, sqrt(n_sel), sqrt(n_sel) * 1.5, n_sel / 3))))
cv_controls <- list(
  "5-fold"  = trainControl(method = "cv", number = 5,  savePredictions = "final"),
  "10-fold" = trainControl(method = "cv", number = 10, savePredictions = "final"),
  "LOOCV"   = trainControl(method = "LOOCV",           savePredictions = "final")
)

results <- list()
holdout <- list()
for (split_name in names(SPLITS)) {
  set.seed(SEED)
  train_idx <- as.vector(createDataPartition(model_df$soc, p = SPLITS[[split_name]], list = FALSE))
  train_df  <- model_df[train_idx, ]
  test_df   <- model_df[-train_idx, ]

  # Holdout Random Forest (mtry = sqrt(p))
  set.seed(SEED)
  rf <- randomForest(x = train_df[, selected], y = train_df$soc, ntree = N_TREE,
                     mtry = floor(sqrt(n_sel)), importance = TRUE)
  holdout[[split_name]] <- list(obs_train  = train_df$soc,
                                pred_train = predict(rf, train_df[, selected]),
                                obs_test   = test_df$soc,
                                pred_test  = predict(rf, test_df[, selected]))
  m_train <- with(holdout[[split_name]], metrics(obs_train, pred_train))
  m_test  <- with(holdout[[split_name]], metrics(obs_test, pred_test))

  # Blend on the same rows
  set.seed(SEED)
  blend   <- fit_blend(train_df[, selected], train_df$soc, xy[train_idx, , drop = FALSE])
  m_blend <- metrics(test_df$soc,
                     predict_blend(blend, test_df[, selected], xy[-train_idx, , drop = FALSE]))

  cat(sprintf("\n%s | train %d | test %d\n", split_name, nrow(train_df), nrow(test_df)))
  cat(sprintf("  Holdout test  R2 %.3f  RMSE %.3f  | blend R2 %.3f  RMSE %.3f\n",
              m_test["R2"], m_test["RMSE"], m_blend["R2"], m_blend["RMSE"]))
  cat(sprintf("  Holdout train R2 %.3f  RMSE %.3f\n", m_train["R2"], m_train["RMSE"]))

  # mtry tuned by each cross-validation scheme (caret selects lowest RMSE)
  for (cv_name in names(cv_controls)) {
    set.seed(SEED)
    fit <- train(soc ~ ., data = train_df, method = "rf", trControl = cv_controls[[cv_name]],
                 tuneGrid = data.frame(mtry = mtry_grid), ntree = N_TREE, importance = TRUE)
    m_cv <- metrics(test_df$soc, predict(fit, test_df))
    top  <- which.max(fit$results$Rsquared)            # best cross-validated R2 on the grid
    results[[length(results) + 1]] <- data.frame(
      Split           = split_name,
      CV              = cv_name,
      Best_mtry       = fit$bestTune$mtry,
      Train_R2        = round(m_train[["R2"]], 3),
      Train_RMSE      = round(m_train[["RMSE"]], 3),
      Test_R2         = round(m_cv[["R2"]], 3),
      Test_RMSE       = round(m_cv[["RMSE"]], 3),
      CV_R2           = round(fit$results$Rsquared[top], 3),
      CV_RMSE         = round(fit$results$RMSE[top], 3),
      Blend_Test_R2   = round(m_blend[["R2"]], 3),     # blend is not mtry-tuned:
      Blend_Test_RMSE = round(m_blend[["RMSE"]], 3)    # same for every CV row of a split
    )
    cat(sprintf("  %-8s mtry %d | test R2 %.3f | CV R2 %.3f\n",
                cv_name, fit$bestTune$mtry, m_cv[["R2"]], fit$results$Rsquared[top]))
  }
}
results_df <- do.call(rbind, results)
write.csv(results_df, out("RF_All_Results.csv"), row.names = FALSE)

# Observed vs predicted (holdout Random Forest)
png(out("Plot_ObsVsPred.png"), width = 2400, height = 2000, res = 200)
grid.arrange(
  scatter_plot(holdout[["80:20"]]$obs_test,  holdout[["80:20"]]$pred_test,  "80:20 - test set"),
  scatter_plot(holdout[["70:30"]]$obs_test,  holdout[["70:30"]]$pred_test,  "70:30 - test set"),
  scatter_plot(holdout[["80:20"]]$obs_train, holdout[["80:20"]]$pred_train, "80:20 - training set"),
  scatter_plot(holdout[["70:30"]]$obs_train, holdout[["70:30"]]$pred_train, "70:30 - training set"),
  ncol = 2, top = "SOC Random Forest - observed vs predicted")
invisible(dev.off())

p <- ggplot(results_df, aes(CV, Test_R2, fill = Split)) +
  geom_col(position = position_dodge(0.75), width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", Test_R2)),
            position = position_dodge(0.75), vjust = -0.4, size = 3) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Random Forest test R² by split and cross-validation", x = NULL, y = "Test R²") +
  theme_soc
ggsave(out("Plot_Model_Comparison.png"), p, width = 10, height = 6, dpi = 200, bg = "white")


# =============================================================================
# 4. BEST MODEL AND FINAL FIT
# =============================================================================
log_step("[4/6] Best model")

best <- results_df[order(-results_df$Test_R2, -results_df$CV_R2), ][1, ]
cat(sprintf("Best: %s split, %s | mtry %d | test R2 %.3f | CV R2 %.3f | blend test R2 %.3f\n",
            best$Split, best$CV, best$Best_mtry, best$Test_R2, best$CV_R2, best$Blend_Test_R2))

# Final Random Forest on all samples
set.seed(SEED)
final_rf <- randomForest(soc ~ ., data = model_df, ntree = N_TREE,
                         mtry = best$Best_mtry, importance = TRUE)
oob_r2   <- 1 - final_rf$mse[N_TREE] / var(model_df$soc)
cat(sprintf("Final RF (all %d samples): OOB R2 %.3f | OOB RMSE %.3f\n",
            nrow(model_df), oob_r2, sqrt(final_rf$mse[N_TREE])))
saveRDS(final_rf, out("SOC_FinalRF_Model.rds"))


# =============================================================================
# 4b. ROBUSTNESS CHECK — repeated 80:20 splits, selection inside training
# =============================================================================
robust <- NULL
if (RUN_ROBUST) {
  log_step(sprintf("[4b] Robustness check: %d repeated 80:20 splits", N_REPEATS))
  robust_one <- function(s) {
    set.seed(s)
    tr  <- as.vector(createDataPartition(full_df$soc, p = 0.8, list = FALSE))
    te  <- setdiff(seq_len(nrow(full_df)), tr)
    sel <- kept_covariates(rank_covariates(full_df[tr, ], covariates))
    set.seed(s)
    rf  <- randomForest(x = full_df[tr, sel], y = full_df$soc[tr], ntree = N_TREE,
                        mtry = floor(sqrt(length(sel))))
    set.seed(s)
    bl  <- fit_blend(full_df[tr, sel], full_df$soc[tr], xy[tr, , drop = FALSE])
    c(Seed          = s,
      RF_Test_R2    = r2(full_df$soc[te], predict(rf, full_df[te, sel])),
      Blend_Test_R2 = r2(full_df$soc[te], predict_blend(bl, full_df[te, sel], xy[te, , drop = FALSE])))
  }
  robust <- as.data.frame(do.call(rbind, par_map(seq_len(N_REPEATS), robust_one)))
  write.csv(robust, out("Robust_Eval.csv"), row.names = FALSE)
  gain <- robust$Blend_Test_R2 - robust$RF_Test_R2
  cat(sprintf("Mean test R2: RF %.3f | blend %.3f | gain %+.3f (SE %.3f) | blend better in %.0f%% of splits\n",
              mean(robust$RF_Test_R2), mean(robust$Blend_Test_R2), mean(gain),
              sd(gain) / sqrt(length(gain)), 100 * mean(gain > 0)))
}


# =============================================================================
# 5. SPATIAL PREDICTION AND MAPS
# =============================================================================
log_step(sprintf("[5/6] Spatial prediction: %s", MODEL_LABEL[[FINAL_MODEL]]))

stack  <- rast(lapply(selected, load_covariate, ref = ref))
pixels <- as.data.frame(stack, na.rm = FALSE)
cells  <- which(complete.cases(pixels))
if (!length(cells)) stop("No raster cell has values for every selected covariate")

if (FINAL_MODEL == "Blend") {
  set.seed(SEED)
  final_blend <- fit_blend(model_df[, selected], model_df$soc, xy)
  saveRDS(final_blend, out("SOC_FinalBlend_Model.rds"))
  pred <- rf_mean_sd(final_blend$rf, cbind(pixels[cells, ], oblique(xyFromCell(stack, cells))))
  pred[, "mean"] <- (pred[, "mean"] + predict(final_blend$sv, as.matrix(pixels[cells, ]))) / 2
} else {
  pred <- rf_mean_sd(final_rf, pixels[cells, ])
}

bnd_file <- list.files(file.path(DATA_DIR, "boundary"), "\\.shp$", full.names = TRUE)[1]
if (is.na(bnd_file)) stop("No .shp in data/boundary/")
boundary <- aggregate(vect(bnd_file))
if (crs(boundary) == "") crs(boundary) <- crs(ref) else boundary <- project(boundary, crs(ref))
boundary_xy <- as.data.frame(st_coordinates(st_cast(st_as_sf(boundary), "MULTILINESTRING")))
boundary_xy$group <- paste(boundary_xy$L1, boundary_xy$L2)
clip <- function(r) mask(crop(r, boundary), boundary)

soc_map <- clip(fill_raster(stack[[1]], cells, pred[, "mean"], "SOC_pct"))
soc_sd  <- clip(fill_raster(stack[[1]], cells, pred[, "sd"],   "SOC_SD"))
zones   <- classify(soc_map, matrix(c(-Inf, 1, 1,  1, 2, 2,  2, 3, 3,  3, 4, 4,  4, Inf, 5),
                                    ncol = 3, byrow = TRUE))
writeRaster(soc_map, out("SOC_Predicted.tif"),   overwrite = TRUE, datatype = "FLT4S")
writeRaster(soc_sd,  out("SOC_Uncertainty.tif"), overwrite = TRUE, datatype = "FLT4S")
writeRaster(zones,   out("SOC_Zones.tif"),       overwrite = TRUE, datatype = "INT1U")

soc_vals   <- values(soc_map, na.rm = TRUE)
area_ha    <- length(soc_vals) * prod(res(soc_map)) / 10000
best_r2    <- if (FINAL_MODEL == "Blend") best$Blend_Test_R2 else best$Test_R2
map_stats  <- c(min = min(soc_vals), max = max(soc_vals), mean = mean(soc_vals),
                median = median(soc_vals), sd = sd(soc_vals))
cat(sprintf("Area %.0f ha | SOC min %.3f | max %.3f | mean %.3f | median %.3f | sd %.3f %%\n",
            area_ha, map_stats["min"], map_stats["max"], map_stats["mean"],
            map_stats["median"], map_stats["sd"]))

subtitle <- sprintf("%s (ntree = %d) | mean %.2f%% | %.0f ha | n = %d | best %s / %s | test R² %.3f",
                    MODEL_LABEL[[FINAL_MODEL]], N_TREE, map_stats["mean"], area_ha,
                    nrow(model_df), best$Split, best$CV, best_r2)
zone_df <- raster_df(zones)
zone_df$value <- factor(zone_df$value, levels = 1:5, labels = ZONES$label)

ggsave(out("SOC_Map.png"), width = 11, height = 9, dpi = 250, bg = "white",
       map_plot(raster_df(soc_map), scale_fill_viridis_c(name = "SOC (%)"),
                "Predicted soil organic carbon - Dhanolti, Uttarakhand", subtitle))
ggsave(out("SOC_Zone_Map.png"), width = 11, height = 9, dpi = 250, bg = "white",
       map_plot(zone_df, scale_fill_manual(values = setNames(ZONES$color, ZONES$label),
                                           name = "SOC class", drop = FALSE),
                "SOC zones - Dhanolti, Uttarakhand", interpolate = FALSE))
ggsave(out("SOC_Uncertainty_Map.png"), width = 11, height = 9, dpi = 250, bg = "white",
       map_plot(raster_df(soc_sd), scale_fill_distiller(palette = "YlOrRd", direction = 1, name = "SD (%)"),
                "SOC prediction uncertainty - Dhanolti",
                sprintf("Standard deviation across %d Random Forest trees", N_TREE)))

write.csv(data.frame(
  Metric = c("Area (ha)", "Min SOC (%)", "Max SOC (%)", "Mean SOC (%)", "Median SOC (%)",
             "Std Dev (%)", "OOB R2 (RF)", "Final Model", "Best Split", "Best CV",
             "Test R2 (RF)", "Test RMSE (RF)", "Test R2 (Blend)", "Test RMSE (Blend)",
             sprintf("Mean Test R2 RF (%d splits)", N_REPEATS),
             sprintf("Mean Test R2 Blend (%d splits)", N_REPEATS),
             "Covariates Used", "N Samples"),
  Value = c(round(area_ha, 1), round(map_stats, 3), round(oob_r2, 3), FINAL_MODEL,
            best$Split, best$CV, best$Test_R2, best$Test_RMSE,
            best$Blend_Test_R2, best$Blend_Test_RMSE,
            if (is.null(robust)) NA else round(mean(robust$RF_Test_R2), 3),
            if (is.null(robust)) NA else round(mean(robust$Blend_Test_R2), 3),
            paste(selected, collapse = "; "), nrow(model_df))
), out("SOC_Map_Summary.csv"), row.names = FALSE)


# =============================================================================
# 6. SPATIAL VALIDATION
#    Random vs spatially blocked 5-fold CV (blocks = k-means clusters of sample
#    locations), with covariate selection inside every training fold. The CV
#    whose test-to-training distances match the map's pixel-to-sample distances
#    is the one that estimates map accuracy (Mila et al. 2022).
# =============================================================================
if (RUN_SPATIAL_CV) {
  log_step(sprintf("[6/6] Spatial validation: random vs spatial-block 5-fold CV (%d repetitions)",
                   N_CV_REPEATS))
  K <- 5
  n <- nrow(full_df)
  schemes <- c("Random", "Spatial blocks")
  folds <- list()
  for (scheme in schemes) for (rep in seq_len(N_CV_REPEATS) - 1) {
    set.seed(rep)
    lab <- if (scheme == "Random") sample(rep_len(1:K, n)) else kmeans(xy, K, nstart = 10)$cluster
    for (b in 1:K) folds[[length(folds) + 1]] <- list(scheme = scheme, rep = rep, b = b,
                                                      test = which(lab == b))
  }

  cv_fold <- function(f) {
    tr  <- setdiff(seq_len(nrow(full_df)), f$test)
    sel <- kept_covariates(rank_covariates(full_df[tr, ], covariates))
    X_tr <- full_df[tr, sel]
    X_te <- full_df[f$test, sel]
    y_tr <- full_df$soc[tr]
    set.seed(f$rep * 10 + f$b)
    p_rf <- predict(randomForest(x = X_tr, y = y_tr, ntree = N_TREE, mtry = floor(sqrt(length(sel)))), X_te)
    p_sv <- predict(fit_svr(X_tr, y_tr), as.matrix(X_te))
    set.seed(f$rep * 10 + f$b)
    p_bl <- predict_blend(fit_blend(X_tr, y_tr, xy[tr, , drop = FALSE]), X_te, xy[f$test, , drop = FALSE])
    nn_dist <- apply(xy[f$test, , drop = FALSE], 1, function(p)
      min(sqrt((xy[tr, 1] - p[1])^2 + (xy[tr, 2] - p[2])^2)))
    list(fold = f, pred = cbind(RF = p_rf, `RF+SVR` = (p_rf + p_sv) / 2, Blend = p_bl),
         nn_dist = nn_dist)
  }
  fold_out <- par_map(folds, cv_fold)

  cv_rows <- list()
  for (scheme in schemes) for (rep in seq_len(N_CV_REPEATS) - 1) {
    reps <- Filter(function(o) o$fold$scheme == scheme && o$fold$rep == rep, fold_out)
    pred <- matrix(NA_real_, n, 3, dimnames = list(NULL, colnames(reps[[1]]$pred)))
    for (o in reps) pred[o$fold$test, ] <- o$pred
    for (m in colnames(pred)) {
      cv_rows[[length(cv_rows) + 1]] <- data.frame(
        CV = paste(scheme, "5-fold"), Rep = rep, Model = m,
        R2 = r2(full_df$soc, pred[, m]), RMSE = sqrt(mean((full_df$soc - pred[, m])^2)))
    }
  }
  cv_summary <- do.call(rbind, cv_rows) %>%
    group_by(CV, Model) %>%
    summarise(Mean_R2 = round(mean(R2), 3), SD_R2 = round(sd(R2), 3),
              Mean_RMSE = round(mean(RMSE), 3), .groups = "drop")

  map_xy   <- xyFromCell(soc_map, which(!is.na(values(soc_map))))
  map_dist <- rep(Inf, nrow(map_xy))
  for (i in seq_len(nrow(xy))) {
    map_dist <- pmin(map_dist, sqrt((map_xy[, 1] - xy[i, 1])^2 + (map_xy[, 2] - xy[i, 2])^2))
  }
  cv_dist <- function(s) unlist(lapply(Filter(function(o) o$fold$scheme == s, fold_out), `[[`, "nn_dist"))
  probs   <- c(0.10, 0.25, 0.50, 0.75, 0.90)
  dist_summary <- round(rbind(
    "CV test -> train (Random)"         = quantile(cv_dist("Random"), probs),
    "CV test -> train (Spatial blocks)" = quantile(cv_dist("Spatial blocks"), probs),
    "Map pixel -> sample"               = quantile(map_dist, probs)))
  colnames(dist_summary) <- paste0("P", probs * 100, " (m)")

  write.csv(cv_summary, out("Spatial_Validation.csv"), row.names = FALSE)
  write.csv(dist_summary, out("Spatial_Validation_Distances.csv"))
  print(as.data.frame(cv_summary), row.names = FALSE)
  cat("\nNearest-neighbour distances (the CV matching the map row estimates map accuracy):\n")
  print(dist_summary)
}

cat("\nDone. Results in", normalizePath(OUT_DIR), "\n")
