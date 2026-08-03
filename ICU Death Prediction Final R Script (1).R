#Load Libraries
library(tidyverse)
library(data.table)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)
library(ggplot2)
library(yardstick)
library(dplyr)
library(ggfortify)
library(ggrepel)
library(tidymodels)
library(knitr)
library(keras3)

#Read Data
train_X <- read_csv("mimic_train_X.csv")
train_y <- read_csv("mimic_train_y.csv")
test_X <- read_csv("mimic_test_X.csv")
secondary_causes <- read_csv("MIMIC_diagnoses.csv")

#Process Secondary Causes
secondary_causes <- secondary_causes |>
  select(SUBJECT_ID, ICD9_CODE) |>
  distinct() |>
  mutate(
    ICD9_CODE = as.character(ICD9_CODE),  
    present = TRUE 
  )

secondary_wide <- secondary_causes |>
  pivot_wider(
    names_from = ICD9_CODE,
    values_from = present,
    values_fill = FALSE,
    names_prefix = "ICD9_"
  ) |>
  mutate(across(-SUBJECT_ID, as.integer)) 

#Process ICD9 diagnosis column in main dataset
diagnosis_wide <- train_X |>
  select(subject_id, ICD9_diagnosis) |>
  distinct() |>
  mutate(
    ICD9_diagnosis = as.character(ICD9_diagnosis),  
    present = TRUE 
  )|>
  pivot_wider(
    names_from = ICD9_diagnosis,
    values_from = present,
    values_fill = list(present = FALSE),                  
    names_prefix = "DIAG_"
  )|>
  mutate(across(-subject_id, as.integer)) 

#Process ICD9_diagnosis column in test set
test_diagnosis_wide <- test_X |>
  select(subject_id, ICD9_diagnosis) |>
  distinct() |> 
  mutate(
    ICD9_diagnosis = as.character(ICD9_diagnosis),  
    present = TRUE 
  )|>
  pivot_wider(
    names_from = ICD9_diagnosis,
    values_from = present,
    values_fill = list(present = FALSE),                  
    names_prefix = "DIAG_"
  )|>
  mutate(across(-subject_id, as.integer))

#Merge secondary causes data, train_x & train_y
secondary_wide <- secondary_wide|>
  rename(subject_id = SUBJECT_ID)

train_X <- train_X |>
  left_join(secondary_wide, by = "subject_id")

train <- bind_cols(train_X, train_y)

combined_data<- train|>
  left_join(diagnosis_wide, by = "subject_id")

#Process overall train data 
full_df <- combined_data|>
  mutate(
    entered_age = as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25,
    entered_age = ifelse(entered_age > 89, 90, entered_age),
    ICD9_diagnosis = as.factor(ICD9_diagnosis),
    HOSPITAL_EXPIRE_FLAG = as.factor(HOSPITAL_EXPIRE_FLAG))|>
  select(-subject_id, -hadm_id, -icustay_id...4, -...1, -DOB, -ADMITTIME, -Diff, -RELIGION, -MARITAL_STATUS, -ETHNICITY, -DIAGNOSIS,-ICD9_diagnosis)

#Process Test Data
test_X <- test_X|>
  mutate(
    entered_age = as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25,
    entered_age = ifelse(entered_age > 89, 90, entered_age))

test_df <- test_X |>
  left_join(secondary_wide, by = "subject_id")|>
  left_join(test_diagnosis_wide, by = "subject_id")

test_df <- test_df|>
  select(-subject_id, -hadm_id, -icustay_id, -DOB, -ADMITTIME, -Diff, -RELIGION, -MARITAL_STATUS, -ETHNICITY, -DIAGNOSIS, -...1)

#Ensure all columns are same in test and train set (excluding hospital expire flag)
expected_vars <- setdiff(names(full_df), "HOSPITAL_EXPIRE_FLAG")

missing_vars <- setdiff(expected_vars, names(test_df))
test_df[missing_vars] <- 0

test_df <- test_df[, expected_vars]

#Best Logistic Model
logit_spec <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

logit_fit_1 <- logit_spec |> 
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = full_df)

test_predictions <- predict(logit_fit_1, new_data = test_df, type = "prob")

results <- test_X |> 
  bind_cols(test_predictions) |> 
  select(icustay_id,.pred_1)

results <- results|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results, "hospital_expire_predictions.csv")

#Decision Tree Model (Does not work)
tune_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) |>
  set_engine("rpart") |>
  set_mode("classification")

tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

set.seed(123)
p_folds <- vfold_cv(full_df, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

tree_wf <- workflow() |>
  add_model(tune_spec) |>
  add_formula(HOSPITAL_EXPIRE_FLAG ~ .)

set.seed(123)
tree_res <- tune_grid(
  tree_wf,
  resamples = p_folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc, accuracy)
)

tree_res |>
  collect_metrics() |>
  slice_head(n = 6) |>
  kable(caption = "First 6 Hyperparameter Tuning Results")

best_tree <- select_best(tree_res, metric = "roc_auc")
final_tree <- finalize_workflow(tree_wf, best_tree)

final_tree_fit <- fit(final_tree, data = full_df)

test_decisiontree <- predict(final_tree_fit, new_data = test_df, type = "prob")

results_dt <- test_X |> 
  bind_cols(test_decisiontree) |> 
  select(icustay_id,.pred_1)

results_dt <- results_dt|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_dt, "decision_tree_predictions.csv")

#Best Random forest Model 0.937
set.seed(234)
rf_spec <- rand_forest(
  mtry = 87,
  trees = 1000
) |> 
  set_mode("classification") |> 
  set_engine("ranger")

rf_fit <- rf_spec |> 
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = full_df)

rf_preds <- predict(rf_fit, test_df, type = "prob") 

results_rf_1 <- test_X |> 
  bind_cols(rf_preds) |> 
  select(icustay_id,.pred_1)

results_rf_1 <- results_rf_1|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_rf_1, "random_forest_1_predictions.csv")

#Random Forest In sample test
rf_preds_in <- predict(rf_fit, full_df, type = "prob")
roc_rf <- roc(full_df$HOSPITAL_EXPIRE_FLAG, rf_preds_in$.pred_1)
print(paste("Random Forest In-Sample AUC:", auc(roc_rf)))

#Best Boosted Tree Model 0.902
xgb_spec <- boost_tree() |> 
  set_engine("xgboost") |> 
  set_mode("classification")

xgb_fit <- xgb_spec |> fit(HOSPITAL_EXPIRE_FLAG ~ ., data = full_df)
test_preds_xgb <- predict(xgb_fit, test_df, type = "prob")

results_bt_1 <- test_X |> 
  bind_cols(test_preds_xgb) |> 
  select(icustay_id,.pred_1)

results_bt_1 <- results_bt_1|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_bt_1, "boosted_tree_predictions.csv")

#Boosted Tree in sample test
xgb_preds_in <- predict(xgb_fit, full_df, type = "prob")
roc_xgb <- roc(full_df$HOSPITAL_EXPIRE_FLAG, xgb_preds_in$.pred_1)
print(paste("Boosted Forest In-Sample AUC:", auc(roc_xgb)))

#Neural Network using secondary Causes [low roc auc]
class_weights <- list("0" = 1, "1" = 4) #due to class imbalance

full_df <- full_df |>
  mutate(
    GENDER = as.factor(GENDER),
    ADMISSION_TYPE = as.factor(ADMISSION_TYPE),
    FIRST_CAREUNIT = as.factor(FIRST_CAREUNIT)
  )

test_df <- test_df |>
  mutate(
    GENDER = factor(GENDER, levels = levels(full_df$GENDER)),
    ADMISSION_TYPE = factor(ADMISSION_TYPE, levels = levels(full_df$ADMISSION_TYPE)),
    FIRST_CAREUNIT = factor(FIRST_CAREUNIT, levels = levels(full_df$FIRST_CAREUNIT))
  )

nn_X <- model.matrix(HOSPITAL_EXPIRE_FLAG ~ . - 1, data = full_df)
nn_X_test <- model.matrix(~ . - 1, data = test_df)

nn_X <- as.matrix(nn_X)
storage.mode(nn_X) <- "double"

nn_X_test <- as.matrix(nn_X_test)
storage.mode(nn_X_test) <- "double"

nn_y <- as.numeric(as.character(full_df$HOSPITAL_EXPIRE_FLAG)) 

nn_model <- keras_model_sequential() |>
  layer_dense(units = 32L, activation = 'relu', input_shape = list(ncol(nn_X))) |>
  layer_dense(units = 16L, activation = 'relu') |>
  layer_dense(units = 1L, activation = 'sigmoid')

nn_model |> compile(
  optimizer = optimizer_adam(learning_rate = 0.0005),
  loss = loss_binary_crossentropy,
  metrics = list("accuracy", metric_auc(name = "auc"))
)

early_stop <- callback_early_stopping(
  monitor = "val_auc",
  patience = 10L,
  mode = "max",
  restore_best_weights = TRUE
)

history <- nn_model |> fit(
  x = nn_X,
  y = nn_y,
  epochs = 100L,
  batch_size = 32L,
  validation_split = 0.2,
  callbacks = list(early_stop),
  verbose = 2L,
  class_weight = class_weights
)

nn_preds <- nn_model |> predict(nn_X_test)

results_nn_2 <- test_X |>
  select(icustay_id) |>
  bind_cols(HOSPITAL_EXPIRE_FLAG = nn_preds[, 1]) |>
  rename(ID = icustay_id)

write_csv(results_nn_2, "neural_network_predictions_2.csv")
#-------------------------------------------------------------------------------
#Data Process for PCA attempt
train_X <- read_csv("mimic_train_X.csv")
train_y <- read_csv("mimic_train_y.csv")
test_X <- read_csv("mimic_test_X.csv")

train <- bind_cols(train_X, train_y)
main_df <- main_df |>
  mutate(
    ICD9_prefix = substr(ICD9_diagnosis, 1, 3),
    ICD9_prefix = as.factor(ICD9_prefix)
  )

# Define top ICD9 prefixes from TRAINING DATA only
top_prefixes <- main_df |>
  count(ICD9_prefix, sort = TRUE) |>
  head(20) |>  
  pull(ICD9_prefix)

main_df <- main_df |>
  mutate(
    ICD9_prefix_grouped = ifelse(ICD9_prefix %in% top_prefixes, 
                                 as.character(ICD9_prefix), 
                                 "Other")
  )

icd9_onehot <- main_df |>
  select(icustay_id, ICD9_prefix_grouped) |>
  mutate(count = 1) |>
  pivot_wider(
    names_from = ICD9_prefix_grouped,
    values_from = count,
    values_fill = list(count = 0)
  )

# Train PCA
pca_res <- prcomp(icd9_onehot[, -1], center = TRUE, scale. = TRUE)
top_pcs <- as.data.frame(pca_res$x[, 1:5])  
top_pcs$icustay_id <- icd9_onehot$icustay_id

# Final training dataset
full_df <- main_df |>
  select(-ICD9_diagnosis) |>
  left_join(top_pcs, by = "icustay_id") |>
  mutate(
    entered_age = as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25,
    entered_age = ifelse(entered_age > 89, 90, entered_age)
  ) |>
  select(
    HeartRate_Min, HeartRate_Max, HeartRate_Mean,
    SysBP_Min, SysBP_Max, SysBP_Mean,
    DiasBP_Min, DiasBP_Max, DiasBP_Mean,
    MeanBP_Min, MeanBP_Max, MeanBP_Mean,
    RespRate_Min, RespRate_Max, RespRate_Mean,
    TempC_Min, TempC_Max, TempC_Mean,
    SpO2_Min, SpO2_Max, SpO2_Mean,
    Glucose_Min, Glucose_Max, Glucose_Mean,
    entered_age, HOSPITAL_EXPIRE_FLAG, ADMISSION_TYPE,
    PC1, PC2, PC3, PC4, PC5
  )

# Prepare test data with same transformations
test_df <- test_X |>
  mutate(
    entered_age = as.numeric(difftime(ADMITTIME, DOB, units = "days")) / 365.25,
    entered_age = ifelse(entered_age > 89, 90, entered_age),
    ICD9_prefix = substr(ICD9_diagnosis, 1, 3),
    ICD9_prefix_grouped = ifelse(ICD9_prefix %in% top_prefixes, 
                                 as.character(ICD9_prefix), 
                                 "Other")
  )

icd9_onehot_test <- test_df |>
  select(icustay_id, ICD9_prefix_grouped) |>
  mutate(count = 1) |>
  pivot_wider(
    names_from = ICD9_prefix_grouped,
    values_from = count,
    values_fill = list(count = 0)
  )

# Ensure test data has same columns as training
missing_cols <- setdiff(names(icd9_onehot)[-1], names(icd9_onehot_test))
for (col in missing_cols) {
  icd9_onehot_test[[col]] <- 0
}

# Reorder columns to match training PCA input
icd9_onehot_test <- icd9_onehot_test |>
  select(all_of(names(icd9_onehot)))

# Apply PCA transform
pca_test_mat <- predict(pca_res, newdata = icd9_onehot_test[, -1])
top_pcs_test <- as.data.frame(pca_test_mat[, 1:5])
top_pcs_test$icustay_id <- icd9_onehot_test$icustay_id

# Final test dataset
test_df <- test_df |>
  left_join(top_pcs_test, by = "icustay_id") |>
  select(
    HeartRate_Min, HeartRate_Max, HeartRate_Mean,
    SysBP_Min, SysBP_Max, SysBP_Mean,
    DiasBP_Min, DiasBP_Max, DiasBP_Mean,
    MeanBP_Min, MeanBP_Max, MeanBP_Mean,
    RespRate_Min, RespRate_Max, RespRate_Mean,
    TempC_Min, TempC_Max, TempC_Mean,
    SpO2_Min, SpO2_Max, SpO2_Mean,
    Glucose_Min, Glucose_Max, Glucose_Mean,
    entered_age, ADMISSION_TYPE,
    PC1, PC2, PC3, PC4, PC5
  )

#Logistic with PCA
logit_spec <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

logit_fit_1 <- logit_spec |> 
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = full_df)


test_predictions <- predict(logit_fit_1, new_data = test_df, type = "prob")

results <- test_df |> 
  bind_cols(test_predictions) |> 
  select(icustay_id,.pred_1)

results <- results|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results, "hospital_expire_predictions.csv")

#Decision Tree
tune_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) |>
  set_engine("rpart") |>
  set_mode("classification")

tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

set.seed(123)
p_folds <- vfold_cv(clean_data_1, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

tree_wf <- workflow()|>
  add_model(tune_spec)|>
  add_formula(HOSPITAL_EXPIRE_FLAG ~ .)

tree_res <- tune_grid(
  tree_wf,
  resamples = p_folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc, accuracy)
)

best_tree <- select_best(tree_res, metric = "roc_auc")
final_tree <- finalize_workflow(tree_wf, best_tree)

final_tree_fit <- fit(final_tree, data = full_df)

test_decisiontree <- predict(final_tree_fit, new_data = test_df, type = "prob")

results_dt <- test_X |> 
  bind_cols(test_decisiontree) |> 
  select(icustay_id,.pred_1)

results_dt <- results_dt|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_dt, "decision_tree_predictions.csv")

#Random Forest with cv & PCA
full_df <- full_df |>
  mutate(HOSPITAL_EXPIRE_FLAG = factor(HOSPITAL_EXPIRE_FLAG, levels = c(0, 1)))

set.seed(123)
p_folds <- vfold_cv(full_df, v = 5, strata = HOSPITAL_EXPIRE_FLAG)

rf_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_wf <- workflow() |>
  add_model(rf_spec) |>
  add_formula(HOSPITAL_EXPIRE_FLAG ~ .)

rf_params <- parameters(rf_spec) |>
  finalize(select(full_df, -HOSPITAL_EXPIRE_FLAG)) 

rf_grid <- grid_random(rf_params, size = 10)

rf_tune <- tune_grid(
  rf_wf,
  resamples = p_folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy)
)

best_rf <- select_best(rf_tune, metric = "roc_auc")

final_rf_wf <- finalize_workflow(rf_wf, best_rf)

final_rf_fit <- fit(final_rf_wf, data = full_df)

rf_probs <- predict(final_rf_fit, new_data = test_df, type = "prob")

results_rf <- test_df |> 
  bind_cols(rf_probs) |> 
  select(icustay_id,.pred_1)

results_rf <- results_rf|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_rf, "random_forest_predictions.csv")

#Random Forest with PCA and no CV
set.seed(123)
rf_spec <- rand_forest(
  mtry = 2,
  trees = 1000
) |> 
  set_mode("classification") |> 
  set_engine("ranger")

rf_fit <- rf_spec |> 
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = full_df)

rf_preds <- predict(rf_fit, test_df, type = "prob") 

results_rf_1 <- test_X |> 
  bind_cols(rf_preds) |> 
  select(icustay_id,.pred_1)

results_rf_1 <- results_rf_1|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_rf_1, "random_forest_1_predictions.csv")

#Boosted Tree PCA without CV
xgb_spec <- boost_tree() |> 
  set_engine("xgboost") |> 
  set_mode("classification")

xgb_fit <- xgb_spec |> fit(HOSPITAL_EXPIRE_FLAG ~ ., data = full_df)
test_preds_xgb <- predict(xgb_fit, test_df, type = "prob")

results_bt_1 <- test_X |> 
  bind_cols(test_preds_xgb) |> 
  select(icustay_id,.pred_1)

results_bt_1 <- results_bt_1|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_bt_1, "boosted_tree_predictions.csv")

#Boosted Tree PCA with CV
boost_spec <- boost_tree(
  trees = 1000,
  learn_rate = tune(),
  tree_depth = tune(),
  loss_reduction = tune()
) |>
  set_engine("xgboost") |>
  set_mode("classification")

boost_wf <- workflow() |>
  add_model(boost_spec) |>
  add_formula(HOSPITAL_EXPIRE_FLAG ~ .)

boost_grid <- grid_random(parameters(boost_spec), size = 10)

boost_tune <- tune_grid(
  boost_wf,
  resamples = p_folds,
  grid = boost_grid,
  metrics = metric_set(roc_auc, accuracy)
)

best_boost <- select_best(boost_tune, "roc_auc")

final_boost <- finalize_workflow(boost_wf, best_boost) %>%
  fit(data = full_df)

boost_predictions <- predict(final_boost, new_data = test_df, type = "prob")

results_bt_1 <- test_X |> 
  bind_cols(boost_predictions) |> 
  select(icustay_id,.pred_1)

results_bt_1 <- results_bt_1|>
  rename( HOSPITAL_EXPIRE_FLAG = .pred_1,
          ID = icustay_id)

write_csv(results_bt_1, "boosted_tree_predictions.csv")

#neural network with PCA
set_random_seed(1234)
nn_X <- full_df |>
  select(where(is.numeric), -HOSPITAL_EXPIRE_FLAG) |>
  mutate(across(everything(), as.numeric))|>
  as.matrix()

nn_y <- full_df$HOSPITAL_EXPIRE_FLAG |> 
  as.integer() |>
  matrix(ncol = 1)

# Test data (ensure same columns as training)
nn_test_X <- test_df |>
  select(any_of(colnames(nn_X))) |>  
  mutate(across(everything(), as.numeric)) |>
  as.matrix()

nn_model <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu", input_shape = ncol(nn_X)) |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 16, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 1, activation = "sigmoid")

nn_model |> compile(
  optimizer = optimizer_adam(),
  loss = loss_binary_crossentropy(),
  metrics = list(
    metric_binary_accuracy(),
    metric_auc(name = "auc")
  )
)

early_stop <- callback_early_stopping(
  monitor = "val_auc",
  patience = 10,
  mode = "max",
  restore_best_weights = TRUE
)

history <- nn_model |> fit(
  x = nn_X,
  y = nn_y,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(early_stop),
  verbose = 2
)

nn_preds <- nn_model$predict(nn_test_X)
nn_preds <- as.data.frame(nn_preds)
colnames(nn_preds) <- "HOSPITAL_EXPIRE_FLAG"

results_nn_1 <- test_X |>
  bind_cols(nn_preds) |>
  select(icustay_id, HOSPITAL_EXPIRE_FLAG)

results_nn_1 <- results_nn_1|>
  rename(ID = icustay_id)

write_csv(results_nn_1, "neural_network_predictions.csv")
