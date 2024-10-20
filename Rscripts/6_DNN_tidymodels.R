##=============================================================================##
## Title: Deep neural network for classification - basic
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: October23, 2024
##=============================================================================##

# DNN using tidymodels

# Packages ----------------------------------------------------------------

library(torch)
library(tabnet)
library(tidyverse)
library(tidymodels)
library(finetune) # to use tuning functions from the new finetune package
library(vip) # to plot feature importances
library(mlbench)

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

## Explore data ----
skimr::skim(pima2)

## Split data ----
set.seed(123)
split_ind <- initial_split(pima2)
pima_train <- training(split_ind)
pima_test <- testing(split_ind)

## Preprocessing ----
pima_rc <- 
  recipe(diabetes~., data = pima_train) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_dummy(all_nominal_predictors())

pima_train_processed <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

pima_test_processed <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = pima_test)

## 10-fold CV ----
set.seed(123)
pima_cv <- vfold_cv(pima_train_processed, v = 10)


# Fit resamples -----------------------------------------------------------

## Specify model ----
dnn_mod <- 
  mod <- tabnet(epochs = 50, batch_size = 128) %>%
  set_engine("torch") %>%
  set_mode("classification")

## Specify workflow ----
dnn_wf <- workflow() %>% 
  add_model(dnn_mod) %>% 
  add_recipe(pima_rc)      

## Fit resamples ----
set.seed(123)
dnn_res <- 
  dnn_wf %>% 
  fit_resamples(pima_cv)

dnn_res %>% 
  collect_metrics()
              

# Fit to training data ----------------------------------------------------

dnn_train <- 
  dnn_wf %>% 
  fit(data = pima_train_processed)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
pima_pred <- 
  pima_test_processed %>% 
  bind_cols(predict(dnn_train, new_data = pima_test_processed)) %>% 
  bind_cols(predict(dnn_train, new_data = pima_test_processed, type = "prob"))

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


# Variable importance -----------------------------------------------------

fit <- extract_fit_parsnip(dnn_train)
vip(fit) + theme_minimal()


# Current model:
# Accuracy: 0.776
# ROC_AUC: 0.855

# Third model:
# Accuracy: 0.791
# ROC_AUC: 0.859

# Second model:
# Accuracy: 0.739
# ROC_AUC: 0.827

# First model:
# Accuracy: 0.724
# ROC_AUC: 0.767
