# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import optuna
from sklearn import datasets, model_selection, metrics
from sklearn.model_selection import train_test_split

method_flag = 2  # 0: LightGBM, 1: XGBoost, 2: scikit-learn

number_of_test_samples = 45  # the number of test samples
fold_number = 2  # "fold_number"-fold cross-validation
fraction_of_validation_samples = 0.2  # fraction of validation samples for early stopping. If early stopping is not required and the number of sub-models is set, please set this to 0 
number_of_sub_models = 500  # (This is active only when fraction_of_validation_samples = 0) The number of sub-models 

# load iris dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

# divide samples into training samples and test samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)
x_train_tmp, x_validation, y_train_tmp, y_validation = train_test_split(x_train, y_train,
                                                                        test_size=fraction_of_validation_samples,
                                                                        random_state=0)
# hyperparameter optimization with optuna and modeling
if method_flag == 0:  # LightGBM
    import lightgbm as lgb

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = lgb.LGBMClassifier(n_estimators=1000)
        model.fit(x_train_tmp, y_train_tmp, eval_set=(x_validation, y_validation),
                  eval_metric='logloss', early_stopping_rounds=100)
        best_n_estimators_in_cv = model.best_iteration_


    def objective(trial):
        param = {
            #            'objective': 'multiclass',
            #            'metric': 'multi_logloss',
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

        model = lgb.LGBMClassifier(**param, n_estimators=best_n_estimators_in_cv)
        estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, y_train, cv=fold_number)
        accuracy = metrics.accuracy_score(y_train, estimated_y_in_cv)
        return 1.0 - accuracy


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        model = lgb.LGBMClassifier(**study.best_params, n_estimators=1000)
        model.fit(x_train_tmp, y_train_tmp, eval_set=(x_validation, y_validation),
                  eval_metric='logloss', early_stopping_rounds=100)
        best_n_estimators = model.best_iteration_
    model = lgb.LGBMClassifier(**study.best_params, n_estimators=best_n_estimators)
elif method_flag == 1:  # XGBoost
    import xgboost as xgb

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = xgb.XGBClassifier(n_estimators=1000)
        model.fit(x_train_tmp, y_train_tmp,
                  eval_set=[(x_validation, y_validation.reshape([len(y_validation), 1]))],
                  eval_metric='mlogloss', early_stopping_rounds=100)
        best_n_estimators_in_cv = model.best_iteration


    def objective(trial):
        param = {
            'silent': 1,
            #            'objective': 'binary:logistic',
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

        model = xgb.XGBClassifier(**param, n_estimators=best_n_estimators_in_cv)
        estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, y_train, cv=fold_number)
        accuracy = metrics.accuracy_score(y_train, estimated_y_in_cv)
        return 1.0 - accuracy


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        model = xgb.XGBClassifier(**study.best_params, n_estimators=1000)
        model.fit(x_train_tmp, y_train_tmp,
                  eval_set=[(x_validation, y_validation.reshape([len(y_validation), 1]))],
                  eval_metric='mlogloss', early_stopping_rounds=100)
        best_n_estimators = model.best_iteration
    model = xgb.XGBRegressor(**study.best_params, n_estimators=best_n_estimators)
elif method_flag == 2:  # scikit-learn
    from sklearn.ensemble import GradientBoostingClassifier

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = GradientBoostingClassifier(n_estimators=1000, validation_fraction=fraction_of_validation_samples,
                                           n_iter_no_change=100)
        model.fit(x_train, y_train)
        best_n_estimators_in_cv = len(model.estimators_)


    def objective(trial):
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 20),
            'max_features': trial.suggest_loguniform('max_features', 0.1, 1.0)
        }

        model = GradientBoostingClassifier(**param, n_estimators=best_n_estimators_in_cv)
        estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, y_train, cv=fold_number)
        accuracy = metrics.accuracy_score(y_train, estimated_y_in_cv)
        return 1.0 - accuracy


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        model = GradientBoostingClassifier(**study.best_params, n_estimators=1000,
                                           validation_fraction=fraction_of_validation_samples, n_iter_no_change=100)
        model.fit(x_train, y_train)
        best_n_estimators = len(model.estimators_)
    model = GradientBoostingClassifier(**study.best_params, n_estimators=best_n_estimators)

model.fit(x_train, y_train)
calculated_y_train = model.predict(x_train)
predicted_y_test = model.predict(x_test)

# confusion matrix for training data
confusion_matrix_train = metrics.confusion_matrix(y_train, calculated_y_train, labels=sorted(set(y_train)))
print('training samples')
print(sorted(set(y_train)))
print(confusion_matrix_train)

# estimated_y in cross-validation
estimated_y_in_cv = model_selection.cross_val_predict(model, x_train, y_train, cv=fold_number)
# confusion matrix in cross-validation
confusion_matrix_train_in_cv = metrics.confusion_matrix(y_train, estimated_y_in_cv, labels=sorted(set(y_train)))
print('training samples in CV')
print(sorted(set(y_train)))
print(confusion_matrix_train_in_cv)

# confusion matrix for test data
confusion_matrix_test = metrics.confusion_matrix(y_test, predicted_y_test, labels=sorted(set(y_train)))
print('')
print('test samples')
print(sorted(set(y_train)))
print(confusion_matrix_test)
