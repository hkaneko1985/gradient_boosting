# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split

method_flag = 0  # 0: LightGBM, 1: XGBoost, 2: scikit-learn

number_of_test_samples = 150  # the number of test samples
fold_number = 2  # "fold_number"-fold cross-validation
fraction_of_validation_samples = 0.2  # fraction of validation samples for early stopping. If early stopping is not required and the number of sub-models is set, please set this to 0 
number_of_sub_models = 500  # The number of sub-models 

# load boston dataset
dataset = pd.read_csv('boston.csv', index_col=0)
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

# divide samples into training samples and test samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()  # autoscaling
x_train_tmp, x_validation, autoscaled_y_train_tmp, autoscaled_y_validation = train_test_split(x_train,
                                                                                              autoscaled_y_train,
                                                                                              test_size=fraction_of_validation_samples,
                                                                                              random_state=0)

# hyperparameter optimization with optuna and modeling6
if method_flag == 0:  # LightGBM
    import lightgbm as lgb

    def objective(trial):
        param = {
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0)
        }

        if param['boosting_type'] == 'dart':
            param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        if param['boosting_type'] == 'goss':
            param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])

        model = lgb.LGBMRegressor(**param, n_estimators=number_of_sub_models)
        estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, autoscaled_y_train, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2 = metrics.r2_score(y_train, estimated_y_in_cv)
        return 1.0 - r2


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    model = lgb.LGBMRegressor(**study.best_params, n_estimators=number_of_sub_models)
elif method_flag == 1:  # XGBoost
    import xgboost as xgb

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = xgb.XGBRegressor(n_estimators=1000)
        model.fit(x_train_tmp, autoscaled_y_train_tmp,
                  eval_set=[(x_validation, pd.DataFrame(autoscaled_y_validation))],
                  eval_metric='rmse', early_stopping_rounds=100)
        best_n_estimators_in_cv = model.best_iteration


    def objective(trial):
        param = {
            'silent': 1,
            'objective': 'reg:linear',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
        }

        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

        model = xgb.XGBRegressor(**param, n_estimators=best_n_estimators_in_cv)
        estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, autoscaled_y_train, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2 = metrics.r2_score(y_train, estimated_y_in_cv)
        return 1.0 - r2


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        model = xgb.XGBRegressor(**study.best_params, n_estimators=1000)
        model.fit(x_train_tmp, autoscaled_y_train_tmp,
                  eval_set=[(x_validation, pd.DataFrame(autoscaled_y_validation))],
                  eval_metric='rmse', early_stopping_rounds=100)
        best_n_estimators = model.best_iteration
    model = xgb.XGBRegressor(**study.best_params, n_estimators=best_n_estimators)
elif method_flag == 2:  # scikit-learn
    from sklearn.ensemble import GradientBoostingRegressor

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = GradientBoostingRegressor(n_estimators=1000, validation_fraction=fraction_of_validation_samples,
                                          n_iter_no_change=100)
        model.fit(x_train, autoscaled_y_train)
        best_n_estimators_in_cv = len(model.estimators_)


    def objective(trial):
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 20),
            'max_features': trial.suggest_loguniform('max_features', 0.1, 1.0)
        }

        model = GradientBoostingRegressor(**param, n_estimators=best_n_estimators_in_cv)
        estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, autoscaled_y_train, cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
        r2 = metrics.r2_score(y_train, estimated_y_in_cv)
        return 1.0 - r2


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        model = GradientBoostingRegressor(**study.best_params, n_estimators=1000,
                                          validation_fraction=fraction_of_validation_samples, n_iter_no_change=100)
        model.fit(x_train, autoscaled_y_train)
        best_n_estimators = len(model.estimators_)
    model = GradientBoostingRegressor(**study.best_params, n_estimators=best_n_estimators)

model.fit(x_train, autoscaled_y_train)
calculated_y_train = model.predict(x_train) * y_train.std() + y_train.mean()
predicted_y_test = model.predict(x_test) * y_train.std() + y_train.mean()

# yy-plot for training data
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, calculated_y_train)
y_max = np.max(np.array([np.array(y_train), calculated_y_train]))
y_min = np.min(np.array([np.array(y_train), calculated_y_train]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()
# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((y_train - calculated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((y_train - calculated_y_train) ** 2) / len(y_train)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(y_train - calculated_y_train)) / len(y_train))))

# estimated_y in cross-validation
estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, autoscaled_y_train, cv=fold_number)
estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
# yy-plot in cross-validation
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, estimated_y_in_cv)
y_max = np.max(np.array([np.array(y_train), estimated_y_in_cv]))
y_min = np.min(np.array([np.array(y_train), estimated_y_in_cv]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in CV')
plt.show()
# r2cv, RMSEcv, MAEcv
print('r2cv: {0}'.format(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSEcv: {0}'.format(float((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)))
print('MAEcv: {0}'.format(float(sum(abs(y_train - estimated_y_in_cv)) / len(y_train))))

# yy-plot for test data
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, predicted_y_test)
y_max = np.max(np.array([np.array(y_test), predicted_y_test]))
y_min = np.min(np.array([np.array(y_test), predicted_y_test]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.show()
# r2p, RMSEp, MAEp
print('r2p: {0}'.format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print('RMSEp: {0}'.format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
