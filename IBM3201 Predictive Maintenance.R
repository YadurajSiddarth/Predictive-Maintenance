# =====================================================================================
# IBM3201 / BDS3404 Final Assessment (Aug 2026)
# Predictive Maintenance: Identifying Potential Industrial Machine Failures Using Data Mining
# Dataset : AI4I 2020 Predictive Maintenance Dataset (UCI ID 601), 10,000 records, synthetic
# Task    : Binary classification of `Machine failure` (No / Yes)
# Method  : CRISP-DM   |   Algorithms: Logistic Regression, Random Forest, XGBoost
# Output  : results/*.csv and figures/*.png (used as evidence in the report)
# By.     : Yaduraj Siddarth
# =====================================================================================

# ---- 0. SETUP -----------------------------------------------------------------------
pkgs <- c("tidyverse", "caret", "randomForest", "xgboost", "pROC", "corrplot", "scales")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

SEED <- 42
CV_FOLDS <- 5
CV_REPEATS <- 3          # lower to 1 for a quick test run
set.seed(SEED)
dir.create("results", showWarnings = FALSE); dir.create("figures", showWarnings = FALSE)
theme_set(theme_minimal(base_size = 12))
save_fig <- function(p, name, w = 7, h = 4.5) ggsave(file.path("figures", name), p, width = w, height = h, dpi = 200)

# ---- 1. DATA UNDERSTANDING ----------------------------------------------------------
# Put ai4i2020.csv next to this script, or let the script download it from UCI.
get_data <- function() {
  if (!file.exists("ai4i2020.csv")) {
    url <- "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
    tmp <- tempfile(fileext = ".zip")
    download.file(url, tmp, mode = "wb"); unzip(tmp, exdir = ".")
  }
  read.csv("ai4i2020.csv", check.names = FALSE, stringsAsFactors = FALSE)
}
raw <- get_data()
stopifnot(ncol(raw) == 14)
names(raw) <- c("UDI", "Product_ID", "Type", "Air_temp_K", "Process_temp_K", "Rot_speed_rpm",
                "Torque_Nm", "Tool_wear_min", "Machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF")
cat("Dimensions:", dim(raw), "\n"); str(raw); print(summary(raw))
cat("Missing values:", sum(is.na(raw)), " | Duplicate rows (excluding IDs):",
    sum(duplicated(raw[, -(1:2)])), "\n")

# Target distribution (class imbalance)
tab <- raw %>% count(Machine_failure) %>% mutate(pct = round(100 * n / sum(n), 2))
print(tab); write.csv(tab, "results/01_class_distribution.csv", row.names = FALSE)
save_fig(ggplot(raw, aes(factor(Machine_failure, labels = c("No failure", "Failure")),
                         fill = factor(Machine_failure))) + geom_bar(show.legend = FALSE) +
           geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
           labs(x = NULL, y = "Records", title = "Target distribution: Machine failure"),
         "01_class_distribution.png", 5, 4)

# Failure-mode check. IMPORTANT: TWF/HDF/PWF/OSF/RNF are the *causes* of the target, so they must NOT be
# used as predictors (target leakage). We only inspect them.
modes <- c("TWF", "HDF", "PWF", "OSF", "RNF")
mode_tab <- tibble(mode = modes, count = colSums(raw[, modes]))
print(mode_tab); write.csv(mode_tab, "results/02_failure_modes.csv", row.names = FALSE)
raw$n_modes <- rowSums(raw[, modes])
cat("\nMachine_failure vs number of active failure-mode flags (check label consistency):\n")
print(table(Machine_failure = raw$Machine_failure, n_modes = raw$n_modes))
save_fig(ggplot(mode_tab, aes(reorder(mode, -count), count)) + geom_col(fill = "#2F5496") +
           geom_text(aes(label = count), vjust = -0.3) +
           labs(x = "Failure mode", y = "Records", title = "Failure modes (inspection only, not used as predictors)"),
         "02_failure_modes.png", 6, 4)

# ---- 2. DATA PREPARATION ------------------------------------------------------------
# 2.1 Cleaning / formatting: drop identifiers and leakage columns; encode target and Type.
# 2.2 Derived features (justified by the documented failure mechanisms of the dataset):
#     Temp_diff   = Process temp - Air temp         (heat-dissipation failure depends on this gap and on speed)
#     Power_W     = Torque * rpm * 2*pi/60          (power failure occurs when power is too low or too high)
#     Wear_torque = Tool wear * Torque              (overstrain failure depends on this product)
df <- raw %>%
  transmute(Machine_failure = factor(Machine_failure, levels = c(0, 1), labels = c("No", "Yes")),
            Type = factor(Type, levels = c("L", "M", "H")),
            Air_temp_K, Process_temp_K, Rot_speed_rpm, Torque_Nm, Tool_wear_min,
            Temp_diff = Process_temp_K - Air_temp_K,
            Power_W = Torque_Nm * Rot_speed_rpm * 2 * pi / 60,
            Wear_torque = Tool_wear_min * Torque_Nm)

base_vars <- c("Type", "Air_temp_K", "Process_temp_K", "Rot_speed_rpm", "Torque_Nm", "Tool_wear_min")
eng_vars  <- c(base_vars, "Temp_diff", "Power_W", "Wear_torque")

# 2.3 Exploration
num_vars <- setdiff(eng_vars, "Type")
long <- df %>% select(Machine_failure, all_of(num_vars)) %>%
  pivot_longer(-Machine_failure, names_to = "variable", values_to = "value")
save_fig(ggplot(long, aes(Machine_failure, value, fill = Machine_failure)) + geom_boxplot(show.legend = FALSE, outlier.size = 0.6) +
           facet_wrap(~variable, scales = "free_y") + labs(x = NULL, y = NULL, title = "Numeric features by failure status"),
         "03_boxplots_by_failure.png", 9, 6)
png("figures/04_correlation_heatmap.png", 1600, 1400, res = 200)
corrplot(cor(df[, num_vars]), method = "color", type = "upper", addCoef.col = "black", number.cex = 0.6,
         tl.col = "black", tl.cex = 0.8, title = "Correlation of numeric features", mar = c(0, 0, 2, 0))
dev.off()
save_fig(ggplot(df, aes(Rot_speed_rpm, Torque_Nm, colour = Machine_failure)) + geom_point(alpha = 0.5, size = 0.9) +
           scale_colour_manual(values = c(No = "grey70", Yes = "#C00000")) +
           labs(x = "Rotational speed [rpm]", y = "Torque [Nm]", title = "Torque vs rotational speed by failure status"),
         "05_torque_vs_speed.png", 7, 5)
write.csv(df %>% group_by(Type) %>% summarise(n = n(), failures = sum(Machine_failure == "Yes"),
                                              failure_rate_pct = round(100 * mean(Machine_failure == "Yes"), 2)),
          "results/03_failure_rate_by_type.csv", row.names = FALSE)

# 2.4 Stratified 70/30 split BEFORE any resampling; imbalance handling is applied inside CV training folds only.
idx   <- createDataPartition(df$Machine_failure, p = 0.70, list = FALSE)
train <- df[idx, ]; test <- df[-idx, ]
cat("\nTrain:", nrow(train), "(failures:", sum(train$Machine_failure == "Yes"), ")",
    "| Test:", nrow(test), "(failures:", sum(test$Machine_failure == "Yes"), ")\n")

# ---- 3. MODELLING -------------------------------------------------------------------
# Same CV folds for every model (paired comparison). Selection metric = F1 of the failure class.
folds <- createMultiFolds(train$Machine_failure, k = CV_FOLDS, times = CV_REPEATS)

f1_summary <- function(data, lev = NULL, model = NULL) {
  
  obs <- factor(as.character(data$obs), levels = c("No", "Yes"))
  pred <- as.character(data$pred)
  
  tp <- sum(obs == "Yes" & pred == "Yes", na.rm = TRUE)
  fp <- sum(obs == "No" & pred == "Yes", na.rm = TRUE)
  fn <- sum(obs == "Yes" & pred == "No", na.rm = TRUE)
  
  prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  rec <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1 <- ifelse(prec + rec == 0, 0,
               2 * prec * rec / (prec + rec))
  
  prob_yes <- suppressWarnings(
    as.numeric(as.character(data$Yes))
  )
  
  class_counts <- table(obs)
  
  if (all(class_counts > 0) &&
      length(prob_yes) == length(obs) &&
      all(is.finite(prob_yes))) {
    
    auc <- as.numeric(
      pROC::auc(
        pROC::roc(
          obs,
          prob_yes,
          levels = c("No", "Yes"),
          direction = "<",
          quiet = TRUE
        )
      )
    )
    
  } else {
    
    warning(
      "AUC skipped: No = ", class_counts["No"],
      ", Yes = ", class_counts["Yes"]
    )
    
    auc <- NA_real_
  }
  
  c(
    F1 = f1,
    Precision = prec,
    Recall = rec,
    AUC = auc
  )
}
make_ctrl <- function(sampling) trainControl(method = "repeatedcv", number = CV_FOLDS, repeats = CV_REPEATS, index = folds,
                                             classProbs = TRUE, summaryFunction = f1_summary,
                                             savePredictions = "final", sampling = sampling, allowParallel = FALSE)
grids <- list(
  LogReg       = NULL,
  RandomForest = expand.grid(mtry = c(2, 4, 6)),
  XGBoost      = expand.grid(nrounds = c(100, 200), max_depth = c(3, 5), eta = 0.1, gamma = 0,
                             colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8))
methods <- c(LogReg = "glm", RandomForest = "rf", XGBoost = "xgbTree")

specs <- expand.grid(algo = names(methods), features = c("Raw", "Engineered"),
                     imbalance = c("None", "Upsampling"), stringsAsFactors = FALSE)
models <- list()
for (i in seq_len(nrow(specs))) {
  s <- specs[i, ]; key <- paste(s$algo, s$features, s$imbalance, sep = " | ")
  vars <- if (s$features == "Raw") base_vars else eng_vars
  cat("Training:", key, "\n"); set.seed(SEED)
  args <- list(form = Machine_failure ~ ., data = train[, c("Machine_failure", vars)], method = methods[[s$algo]],
               metric = "F1", maximize = TRUE, preProcess = c("center", "scale"),
               trControl = make_ctrl(if (s$imbalance == "Upsampling") "up" else NULL))
  if (!is.null(grids[[s$algo]])) args$tuneGrid <- grids[[s$algo]]
  if (s$algo == "LogReg") args$family <- binomial()
  if (s$algo == "RandomForest") args$ntree <- 300
  if (s$algo == "XGBoost") args$verbosity <- 0
  models[[key]] <- do.call(caret::train, args)
}
saveRDS(models, "results/models.rds")

# ---- 4. EVALUATION ------------------------------------------------------------------
# 4.1 Cross-validation (training data)
cv_tab <- map_dfr(names(models), function(k) {
  
  r <- models[[k]]$resample
  p <- strsplit(k, " \\| ")[[1]]
  
  tibble(
    Algorithm = p[1],
    Features = p[2],
    Imbalance = p[3],
    F1 = mean(r$F1),
    F1_sd = sd(r$F1),
    Precision = mean(r$Precision),
    Recall = mean(r$Recall),
    Recall_sd = sd(r$Recall),
    AUC = mean(r$AUC, na.rm = TRUE)
  )
  
}) %>% arrange(desc(F1))

print(as.data.frame(cv_tab), digits = 3)

write.csv(
  cv_tab,
  "results/04_cv_results.csv",
  row.names = FALSE
)

# 4.2 Hold-out test set (default 0.5 threshold)
avg_precision <- function(obs, score) {           # area under the precision-recall curve (average precision)
  o <- order(score, decreasing = TRUE); y <- as.integer(obs[o] == "Yes")
  tp <- cumsum(y); prec <- tp / seq_along(y); sum(prec * y) / sum(y)
}
metrics_at <- function(obs, prob, thr = 0.5) {
  pred <- factor(ifelse(prob >= thr, "Yes", "No"), levels = c("No", "Yes"))
  tp <- sum(obs == "Yes" & pred == "Yes"); fp <- sum(obs == "No" & pred == "Yes")
  fn <- sum(obs == "Yes" & pred == "No");  tn <- sum(obs == "No" & pred == "No")
  prec <- ifelse(tp + fp == 0, NA, tp / (tp + fp)); rec <- tp / (tp + fn)
  tibble(TP = tp, FP = fp, FN = fn, TN = tn, Accuracy = (tp + tn) / length(obs), Precision = prec, Recall = rec,
         Specificity = tn / (tn + fp), F1 = ifelse(is.na(prec) | prec + rec == 0, 0, 2 * prec * rec / (prec + rec)),
         MCC = (tp * tn - fp * fn) / sqrt(as.numeric(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
}
test_probs <- map(models, ~ predict(.x, newdata = test, type = "prob")[, "Yes"])
test_tab <- map_dfr(names(models), function(k) {
  p <- strsplit(k, " \\| ")[[1]]
  bind_cols(tibble(Algorithm = p[1], Features = p[2], Imbalance = p[3]), metrics_at(test$Machine_failure, test_probs[[k]]),
            tibble(ROC_AUC = as.numeric(pROC::auc(pROC::roc(test$Machine_failure, test_probs[[k]], levels = c("No", "Yes"),
                                                             direction = "<", quiet = TRUE))),
                   PR_AUC = avg_precision(test$Machine_failure, test_probs[[k]])))
}) %>% arrange(desc(F1))
print(as.data.frame(test_tab), digits = 3); write.csv(test_tab, "results/05_test_results.csv", row.names = FALSE)

# 4.3 Final model: best CV F1 among the three algorithms (one configuration each), threshold tuned on out-of-fold
#     training predictions only (never on the test set).
best_per_algo <- cv_tab %>% group_by(Algorithm) %>% slice_max(F1, n = 1, with_ties = FALSE) %>% ungroup()
write.csv(best_per_algo, "results/06_best_config_per_algorithm.csv", row.names = FALSE)
final_row <- best_per_algo %>% slice_max(F1, n = 1, with_ties = FALSE)
final_key <- paste(final_row$Algorithm, final_row$Features, final_row$Imbalance, sep = " | ")
cat("\nFinal model:", final_key, "\n")
oof <- models[[final_key]]$pred
bt  <- models[[final_key]]$bestTune                 # keep only out-of-fold predictions of the selected tuning values
if (!is.null(bt) && ncol(bt) > 0) {
  keep <- rep(TRUE, nrow(oof))
  for (n in names(bt)) keep <- keep & (oof[[n]] == bt[[n]])
  oof <- oof[keep, ]
}
thr_grid <- seq(0.05, 0.95, by = 0.01)
thr_tab <- map_dfr(thr_grid, function(t) bind_cols(tibble(threshold = t), metrics_at(oof$obs, oof$Yes, t)))
best_thr <- thr_tab$threshold[which.max(thr_tab$F1)]
cat("Threshold maximising out-of-fold F1:", best_thr, "\n")
write.csv(thr_tab, "results/07_threshold_curve_oof.csv", row.names = FALSE)
final_default <- metrics_at(test$Machine_failure, test_probs[[final_key]], 0.5) %>% mutate(threshold = 0.5)
final_tuned   <- metrics_at(test$Machine_failure, test_probs[[final_key]], best_thr) %>% mutate(threshold = best_thr)
final_tab <- bind_rows(final_default, final_tuned); print(as.data.frame(final_tab), digits = 3)
write.csv(final_tab, "results/08_final_model_test.csv", row.names = FALSE)

# 4.4 Uncertainty: bootstrap 95% CI for test Recall and F1 of the final model (only ~100 failures in the test set)
set.seed(SEED)
boot <- map_dfr(1:1000, function(b) { i <- sample(nrow(test), replace = TRUE)
  m <- metrics_at(test$Machine_failure[i], test_probs[[final_key]][i], best_thr); tibble(Recall = m$Recall, F1 = m$F1, Precision = m$Precision) })
ci <- boot %>% summarise(across(everything(), list(lo = ~quantile(.x, .025, na.rm = TRUE), hi = ~quantile(.x, .975, na.rm = TRUE))))
print(ci); write.csv(ci, "results/09_bootstrap_ci_final_model.csv", row.names = FALSE)

# 4.5 Paired comparison on CV folds: best configuration of each algorithm vs Logistic Regression baseline
keys_best <- best_per_algo %>% mutate(key = paste(Algorithm, Features, Imbalance, sep = " | ")) %>% select(Algorithm, key)
fold_f1 <- map(setNames(keys_best$key, keys_best$Algorithm), ~ models[[.x]]$resample %>% arrange(Resample) %>% pull(F1))
cmp <- map_dfr(c("RandomForest", "XGBoost"), function(a) {
  d <- fold_f1[[a]] - fold_f1[["LogReg"]]; n <- length(d); k <- CV_FOLDS * CV_REPEATS
  # Nadeau & Bengio (2003) corrected resampled t-test; test/train size ratio = 1/(k_folds - 1)
  tstat <- mean(d) / sqrt((1 / n + (1 / (CV_FOLDS - 1))) * var(d))
  tibble(comparison = paste(a, "vs LogReg (CV F1)"), mean_diff = mean(d), corrected_t_p = 2 * pt(-abs(tstat), n - 1))
})
print(cmp); write.csv(cmp, "results/10_paired_tests.csv", row.names = FALSE)

# ---- 5. VISUALISATIONS OF RESULTS ---------------------------------------------------
curve_keys <- best_per_algo %>% mutate(key = paste(Algorithm, Features, Imbalance, sep = " | "))
roc_df <- map_dfr(seq_len(nrow(curve_keys)), function(i) {
  r <- pROC::roc(test$Machine_failure, test_probs[[curve_keys$key[i]]], levels = c("No", "Yes"), direction = "<", quiet = TRUE)
  tibble(Algorithm = curve_keys$Algorithm[i], FPR = 1 - r$specificities, TPR = r$sensitivities)
})
save_fig(ggplot(roc_df, aes(FPR, TPR, colour = Algorithm)) + geom_line(linewidth = 1) + geom_abline(linetype = 2, colour = "grey50") +
           labs(title = "ROC curves (hold-out test set)", x = "False positive rate", y = "True positive rate"), "06_roc_curves.png")
pr_df <- map_dfr(seq_len(nrow(curve_keys)), function(i) {
  s <- test_probs[[curve_keys$key[i]]]; o <- order(s, decreasing = TRUE); y <- as.integer(test$Machine_failure[o] == "Yes")
  tibble(Algorithm = curve_keys$Algorithm[i], Recall = cumsum(y) / sum(y), Precision = cumsum(y) / seq_along(y))
})
save_fig(ggplot(pr_df, aes(Recall, Precision, colour = Algorithm)) + geom_line(linewidth = 1) +
           geom_hline(yintercept = mean(test$Machine_failure == "Yes"), linetype = 2, colour = "grey50") +
           labs(title = "Precision-recall curves (hold-out test set)"), "07_pr_curves.png")

cm_df <- map_dfr(seq_len(nrow(curve_keys)), function(i) {
  pred <- factor(ifelse(test_probs[[curve_keys$key[i]]] >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
  as.data.frame(table(Actual = test$Machine_failure, Predicted = pred)) %>% mutate(Algorithm = curve_keys$Algorithm[i])
})
save_fig(ggplot(cm_df, aes(Predicted, Actual, fill = Freq)) + geom_tile(colour = "white") + geom_text(aes(label = Freq), size = 5) +
           scale_fill_gradient(low = "white", high = "#2F5496", guide = "none") + facet_wrap(~Algorithm) +
           labs(title = "Confusion matrices (threshold 0.5, hold-out test set)"), "08_confusion_matrices.png", 10, 3.8)

save_fig(ggplot(thr_tab, aes(threshold)) + geom_line(aes(y = Precision, colour = "Precision"), linewidth = 1) +
           geom_line(aes(y = Recall, colour = "Recall"), linewidth = 1) + geom_line(aes(y = F1, colour = "F1"), linewidth = 1) +
           geom_vline(xintercept = best_thr, linetype = 2) + labs(colour = NULL, y = "Score", x = "Decision threshold",
                                                                   title = paste("Threshold trade-off (out-of-fold),", final_key)),
         "09_threshold_tradeoff.png", 8, 4.5)

cv_long <- map_dfr(seq_len(nrow(curve_keys)), function(i) models[[curve_keys$key[i]]]$resample %>%
                     transmute(Algorithm = curve_keys$Algorithm[i], F1, Recall))
save_fig(ggplot(cv_long %>% pivot_longer(c(F1, Recall)), aes(Algorithm, value, fill = Algorithm)) + geom_boxplot(show.legend = FALSE) +
           facet_wrap(~name) + labs(x = NULL, y = NULL, title = "Cross-validation stability (best configuration per algorithm)"),
         "10_cv_stability.png", 8, 4.5)

imp_one <- function(algo) { m <- models[[curve_keys$key[curve_keys$Algorithm == algo]]]
  v <- varImp(m, scale = TRUE)$importance; tibble(feature = rownames(v), importance = v[[1]], Algorithm = algo) }
imp <- bind_rows(imp_one("RandomForest"), imp_one("XGBoost"), imp_one("LogReg"))
write.csv(imp, "results/11_variable_importance.csv", row.names = FALSE)
save_fig(ggplot(imp, aes(reorder(feature, importance), importance, fill = Algorithm)) + geom_col(show.legend = FALSE) + coord_flip() +
           facet_wrap(~Algorithm, scales = "free") + labs(x = NULL, y = "Scaled importance (0-100)", title = "Variable importance"),
         "11_variable_importance.png", 11, 4.5)

# Ablation view: effect of derived features and of upsampling on CV F1 / Recall
save_fig(ggplot(cv_tab, aes(Algorithm, F1, fill = interaction(Features, Imbalance, sep = " + "))) +
           geom_col(position = position_dodge(0.85), width = 0.8) + labs(fill = NULL, title = "Effect of derived features and upsampling (CV F1)") +
           coord_cartesian(ylim = c(0, 1)), "12_ablation_cv_f1.png", 9, 4.5)

writeLines(capture.output(sessionInfo()), "results/sessionInfo.txt")
cat("\nDone. Tables are in results/, figures in figures/.\n")
