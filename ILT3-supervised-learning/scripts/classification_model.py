import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, 
    HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
)
from sklearn.naive_bayes import (
    BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, 
    RidgeClassifier, RidgeClassifierCV, SGDClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore")

# Loading csv
initial_df = pd.read_csv('https://drive.google.com/uc?id=1bK-2cex1c52grsOgijhYWajCdJwXRupT')
submission_df = pd.read_csv('https://drive.google.com/uc?id=13bAAy9An6Tqs6l2vx8sALhMYaL8MCWa8')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# One hot encoding categorical features
initial_with_dummies_df = pd.get_dummies(initial_df[features + ['Survived']])
submission_with_dummies_df = pd.get_dummies(submission_df[features])

X = initial_with_dummies_df.drop(['Survived'], axis=1).values
y = initial_with_dummies_df['Survived'].values
X_submission = submission_with_dummies_df.values

# Data preprocessing - Dividing Data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=1)

# Setup default parameters
estimator = RandomForestClassifier(random_state=1).fit(X_train, y_train)
estimator_kn = LogisticRegression().fit(X_train, y_train)
estimators = [('rf', estimator), ('kn', estimator_kn)]

model_names = [
    'AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB', 'CalibratedClassifierCV',
    ('CategoricalNB', '(min_categories=75)'), ('ClassifierChain', f'(base_estimator={estimator_kn})'),
    'ComplementNB', 'DecisionTreeClassifier', ('DummyClassifier', '(strategy="most_frequent", random_state=1)'),
    'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'GaussianProcessClassifier',
    'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'KNeighborsClassifier',
    'LabelPropagation', 'LabelSpreading', 'LinearDiscriminantAnalysis', 'LinearSVC',
    'LogisticRegression', 'LogisticRegressionCV', 'MLPClassifier', ('MultiOutputClassifier', f'(estimator={estimator})'),
    'MultinomialNB', 'NearestCentroid', 'NuSVC', ('OneVsOneClassifier', f'(estimator={estimator})'),
    ('OneVsRestClassifier', f'(estimator={estimator})'), ('OutputCodeClassifier', f'(estimator={estimator})'),
    'PassiveAggressiveClassifier', ('Perceptron', '(shuffle=False, random_state=1)'),
    'QuadraticDiscriminantAnalysis', ('RadiusNeighborsClassifier', '(radius=50)'),
    'RandomForestClassifier', 'RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier', 'SVC',
    ('StackingClassifier', f'(estimators={estimators})'), ('VotingClassifier', f'(estimators={estimators})')
]

# Useful functions for processing all models
def model_score(model_name, save_to_csv=False):
    if isinstance(model_name, tuple):
        model = eval(model_name[0] + model_name[1])
        name = model_name[0]
    else:
        try:
            model = eval(model_name + '(random_state=1)')
        except:
            model = eval(model_name + '()')
        name = model_name

    if name in ['ClassifierChain', 'MultiOutputClassifier']:
        return model_scores_special(model, save_to_csv=save_to_csv)

    model.fit(X_train, y_train)
    y_final = model.predict(X_submission)
    final_df = pd.DataFrame({'PassengerId': submission_df['PassengerId'], 'Survived': y_final})
    if save_to_csv:
        final_df.to_csv(f'{name}.csv', index=False)

    cv_test = cross_val_score(model, X_test, y_test, cv=5).mean()
    model_score = model.score(X_test, y_test)
    return [name, cv_test, model_score]

def model_scores_special(model, save_to_csv=False):
    model.fit(X_train, y_train.reshape(-1, 1))
    y_final = model.predict(X_submission).astype(int).reshape(-1,)
    final_df = pd.DataFrame({'PassengerId': submission_df['PassengerId'], 'Survived': y_final})
    if save_to_csv:
        final_df.to_csv(f'{str(model).split("(")[0]}.csv', index=False)

    cv_test = cross_val_score(model, X_test, y_test.reshape(-1, 1), cv=5).mean()
    model_score = model.score(X_test, y_test.reshape(-1, 1))
    return [str(model).split('(')[0], cv_test, model_score]

models_scores = [model_score(name) for name in model_names]

models_scores_df = pd.DataFrame(models_scores, columns=['Model', 'Cross_Validation', 'Accuracy'])
models_scores_df.sort_values(['Cross_Validation', 'Accuracy'], ascending=False, inplace=True)
models_scores_df.set_index('Model', inplace=True)

models_scores_df['Real_Score'] = [
    0.76076, 0.77511, 0.77511, 0.77511, 0.77511, 0.77511, 0.77751, 0.77751, 0.72966, 0.76315, 
    0.78229, 0.76555, 0.74401, 0.77033, 0.76794, 0.76794, 0.76794, 0.76794, 0.76794, 0.76076, 
    0.76555, 0.77033, 0.74641, 0.70574, 0.63636, 0.70095, 0.66985, 0.74641, 0.6244, 0.63397, 
    0.65789, 0.66507, 0.6555, 0.66267, 0.62679, 0.65311, 0.64832, 0.61483, 0.61483, 0.61961, 0.622
]

models_scores_df = models_scores_df.astype(float)
models_scores_df['diff_acc'] = models_scores_df['Accuracy'] - models_scores_df['Real_Score']
models_scores_df['diff_CV'] = models_scores_df['Cross_Validation'] - models_scores_df['Real_Score']