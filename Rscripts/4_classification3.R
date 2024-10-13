##=============================================================================##
## Title: Classification model - intermediate2
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: October23, 2024
##=============================================================================##

# Compare several models

# Install packages --------------------------------------------------------
# install.packages("klaR")
# install.packages("mda")
# install.packages("earth")
# install.packages("discrim")


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
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# MARS
mars_spec <- 
  discrim_flexible(prod_degree = tune()) %>% 
  set_engine("earth")

# Regularised discriminant analysis
reg_sepc <- 
  discrim_regularized(frac_common_cov = tune(), frac_identity = tune()) %>% 
  set_engine("klaR")

## Specify workflow ----
all_workflows <- 
  workflow_set(
    preproc = list("formula" = diabetes~.),
    models = list(regularized = reg_sepc, 
                  mars = mars_spec, 
                  cart = dt_spec)
    ) %>% 
  option_add(id = "formula_cart", 
             control = control_grid(extract = function(x) x))
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
da_results <- 
  all_workflows %>% 
  extract_workflow_set_result("formula_regularized")
da_results

## Explore best models ----
autoplot(da_results) + theme_light()
da_results %>% collect_metrics()

da_results %>% show_best(metric = "accuracy")
da_results %>% show_best(metric = "roc_auc")

best_tune <- 
  da_results %>% 
  select_best(metric = "roc_auc")

## Extract worflow ----
da_workflow <- 
  all_workflows %>% 
  extract_workflow("formula_regularized")

## Finalize workflow ----
da_wf_final <- 
  da_workflow %>% 
  finalize_workflow(best_tune)


# Re-fit on training data -------------------------------------------------

da_train <- 
  da_wf_final %>% 
  fit(data = pima_train_process)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
pima_pred <- 
  pima_test_process %>% 
  bind_cols(predict(da_train, new_data = pima_test_process)) %>% 
  bind_cols(predict(da_train, new_data = pima_test_process, type = "prob"))

## Performance metrics ----
## Accuracy
pima_pred %>% 
  accuracy(truth = diabetes, estimate = .pred_class)

## Plot ROC
pima_pred %>% 
  roc_curve(diabetes, .pred_pos) %>% 
  autoplot()

pima_pred %>% 
  roc_auc(diabetes, .pred_pos)

# Remember, we select the best model based on ROC-AUC:

# Current model:
# Accuracy: 0.724
# ROC_AUC: 0.840

# Second model:
# Accuracy: 0.739
# ROC_AUC: 0.827

# First model:
# Accuracy: 0.724
# ROC_AUC: 0.767

