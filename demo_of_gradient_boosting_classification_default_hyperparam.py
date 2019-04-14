# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

from sklearn import datasets, model_selection, metrics
from sklearn.model_selection import train_test_split

method_flag = 0  # 0: LightGBM, 1: XGBoost, 2: scikit-learn

fold_number = 5  # "fold_number"-fold cross-validation
number_of_test_samples = 45  # the number of test samples

# load iris dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

# divide samples into training samples and test samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# modeling
if method_flag == 0:  # LightGBM
    import lightgbm as lgb

    model = lgb.LGBMClassifier()
elif method_flag == 1:  # XGBoost
    import xgboost as xgb

    model = xgb.XGBClassifier()
elif method_flag == 2:  # scikit-learn
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier()

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
