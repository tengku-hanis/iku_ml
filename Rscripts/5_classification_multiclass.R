##=============================================================================##
## Title: Classification model - intermediate3
## Author: Tengku Muhammad Hanis Mokhtar, PhD
## Date: October23, 2024
##=============================================================================##

# Multiclass classification 

# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)


# Data --------------------------------------------------------------------

student_data <- 
  readr::read_delim("Data/student_data.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE) %>% 
  janitor::clean_names()

## Balanced data 
set.seed(123)
student_data2 <- 
  student_data %>% 
  filter(target == "Dropout") %>% 
  slice_sample(n = 794) %>%
  bind_rows(
    student_data %>% 
      filter(target == "Graduate") %>% 
      slice_sample(n = 794)
    ) %>% 
  bind_rows(
    student_data %>% 
      filter(target == "Enrolled")
  ) %>% 
  mutate(target = as.factor(target))

## Explore data ----
skimr::skim(student_data2)

# Make sure "Dropout" is a reference 
levels(student_data2$target)

## Split data ----
set.seed(123)
split_ind <- initial_split(student_data2)
student_data_train <- training(split_ind)
student_data_test <- testing(split_ind)

## Preprocessing ----
data_rc <- 
  recipe(target~., data = student_data_train) 

student_data_train_process <- 
  data_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

student_data_test_process <- 
  data_rc %>% 
  prep() %>% 
  bake(new_data = student_data_test)

## 10-fold CV ----
set.seed(123)
student_cv <- vfold_cv(student_data_train_process, v = 10)


# Tuning ------------------------------------------------------------------

## Specify model ----
dt_mod <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

## Specify workflow ----
dt_wf <- workflow() %>% 
  add_model(dt_mod) %>% 
  add_recipe(data_rc)

## tune_grid ----
set.seed(123)

ctrl <- control_resamples(save_pred = TRUE)
dt_tune <- 
  dt_wf %>% 
  tune_grid(resamples = student_cv,
            grid = 10,
            control = ctrl)

## Explore tuning result ----
autoplot(dt_tune) + theme_light()
dt_tune %>% collect_metrics() 

dt_tune %>% show_best(metric = "accuracy")
dt_tune %>% show_best(metric = "roc_auc") #Hand-Till method

dt_tune %>% 
  collect_predictions() %>% 
  group_by(id) %>% # tree_depth also can
  roc_curve(target, .pred_Dropout, .pred_Enrolled, .pred_Graduate) %>% 
  autoplot()

best_tune <- 
  dt_tune %>% 
  select_best(metric = "roc_auc")

## Finalize workflow ----
dt_wf_final <- 
  dt_wf %>% 
  finalize_workflow(best_tune)


# Re-fit on training data -------------------------------------------------

dt_train <- 
  dt_wf_final %>% 
  fit(data = student_data_train_process)

# Visualise (not practical)
dt_fit <- 
  dt_train %>% 
  extract_fit_parsnip()
rpart.plot::rpart.plot(dt_fit$fit, roundint=FALSE)

# Another way to visualise
vip::vip(dt_fit)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
student_pred <- 
  student_data_test_process %>% 
  bind_cols(predict(dt_train, new_data = student_data_test_process)) %>% 
  bind_cols(predict(dt_train, new_data = student_data_test_process, type = "prob"))

## Performance metrics ----
# 1) custom metric set to evaluate performance
test_performance <- metric_set(accuracy, sens, spec, precision, recall, f_meas) #f_meas = F1-score
test_performance(student_pred, truth = target, estimate = .pred_class)

# 2) Specific metrics
## Accuracy
student_pred %>% 
  accuracy(truth = target, estimate = .pred_class)

## Plot ROC
student_pred %>% 
  roc_curve(target, .pred_Dropout, .pred_Enrolled, .pred_Graduate) %>% 
  autoplot()

student_pred %>% 
  roc_auc(target, .pred_Dropout, .pred_Enrolled, .pred_Graduate)

## Sensitivity based on micro and macro averaging
student_pred %>% 
  sensitivity(truth = target, estimate = .pred_class, estimator = "micro")
student_pred %>% 
  sensitivity(truth = target, estimate = .pred_class, estimator = "macro")

## Sensitivity - change event level
student_pred %>% 
  sensitivity(truth = target, estimate = .pred_class, estimator = "micro",
              event_level = "Dropout")
student_pred %>% 
  sensitivity(truth = target, estimate = .pred_class, estimator = "micro",
              event_level = "Enrolled")
student_pred %>% 
  sensitivity(truth = target, estimate = .pred_class, estimator = "micro",
              event_level = "Graduated")

# 3) Confusion matrix
conf_mat(student_pred, truth = target, estimate = .pred_class) %>% 
  autoplot("heatmap")

# 4) All available metrics
conf_mat(student_pred, truth = target, estimate = .pred_class) %>% 
  summary()

