##=============================================================================##
## Title: Classification model - intermediate2
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: October23, 2024
##=============================================================================##

# Compare several models

# Install packages --------------------------------------------------------
# install.packages("randomForest")
# install.packages("mda")
# install.packages("earth")


# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(mlbench)
library(discrim)


# Data --------------------------------------------------------------------

data("PimaIndiansDiabetes2")
pima <- PimaIndiansDiabetes2

## Balanced data 
set.seed(123)
pima2 <- 
  pima %>% 
  filter(diabetes == "neg") %>% 
  slice_sample(n = 268) %>% 
  bind_rows(
    pima %>% 
      filter(diabetes == "pos")
  ) %>% 
  mutate(diabetes = relevel(diabetes, ref = "pos"))

## Split data ----
set.seed(123)
split_ind <- initial_split(pima2)
pima_train <- training(split_ind)
pima_test <- testing(split_ind)

## Preprocessing ----
pima_rc <- 
  recipe(diabetes~., data = pima_train) %>% 
  step_impute_knn(all_predictors())

pima_train_process <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

pima_test_process <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = pima_test)

## 10-fold CV ----
set.seed(123)
pima_cv <- vfold_cv(pima_train_process, v = 10)


# Tuning ------------------------------------------------------------------

## Specify model ----

# Decision tree
dt_spec <- 
  decision_tree(cost_complexity = tune(), tree_depth = tune(), min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# MARS
mars_spec <- 
  discrim_flexible(prod_degree = tune()) %>% 
  set_engine("earth")

# Random forest
rf_spec <- 
  rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%  
  set_engine("randomForest") %>% 
  set_mode("classification")

## Specify workflow ----
all_workflows <- 
  workflow_set(preproc = list("formula" = diabetes~.),
               models = list(dt_spec, mars_spec, rf_spec)) 
all_workflows

## tune_grid ----
set.seed(123)
all_workflows <- 
  all_workflows %>% 
  workflow_map(resamples = pima_cv, grid = 20, verbose = TRUE)

## Explore tuning result ----
all_workflows
rank_results(all_workflows, rank_metric = "roc_auc")

autoplot(all_workflows, metric = "roc_auc")

## Extract best model ----
best_model <- 
  all_workflows %>% 
  extract_workflow_set_result("formula_discrim_flexible")
best_model

## Explore best models ----
autoplot(best_model) + theme_light()
best_model %>% collect_metrics()

best_model %>% show_best(metric = "accuracy")
best_model %>% show_best(metric = "roc_auc")

best_tune <- 
  best_model %>% 
  select_best(metric = "roc_auc")

## Extract workflow ----
best_model_workflow <- 
  all_workflows %>% 
  extract_workflow("formula_discrim_flexible")

## Finalize workflow ----
wf_final <- 
  best_model_workflow %>% 
  finalize_workflow(best_tune)


# Re-fit on training data -------------------------------------------------

mod_train <- 
  wf_final %>% 
  fit(data = pima_train_process)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
pima_pred <- 
  pima_test_process %>% 
  bind_cols(predict(mod_train, new_data = pima_test_process)) %>% 
  bind_cols(predict(mod_train, new_data = pima_test_process, type = "prob"))

## Performance metrics ----
## Accuracy
pima_pred %>% 
  yardstick::accuracy(truth = diabetes, estimate = .pred_class)

## Plot ROC
pima_pred %>% 
  roc_curve(diabetes, .pred_pos) %>% 
  autoplot()

pima_pred %>% 
  roc_auc(diabetes, .pred_pos)

# Remember, we select the best model based on ROC-AUC:

# Current model:
# Accuracy: 0.791
# ROC_AUC: 0.859

# Second model:
# Accuracy: 0.739
# ROC_AUC: 0.827

# First model:
# Accuracy: 0.724
# ROC_AUC: 0.767

