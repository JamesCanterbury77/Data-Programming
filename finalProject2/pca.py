# from IPython.core.display_functions import display
from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.decomposition import TruncatedSVD


def createList(x, y):
    return [item for item in range(x, y + 1)]


def main():
    df = pd.read_csv('roatan_train.csv')
    train_messages = df['message'].to_list()

    # dfvalid = pd.read_csv('roatan_valid.csv')
    # valid_messages = df['message'].to_list()

    # Train_test_split
    X1, X2, y1, y2 = train_test_split(train_messages, df['Coding:Level1'], random_state=0, train_size=0.7)

    params = []
    scores = []
    models = []

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), KNeighborsClassifier())
    param_grid = {
        "truncatedsvd__n_components": [300, 500],
        'truncatedsvd__n_iter': [20],
        'truncatedsvd__random_state': [0],
        'truncatedsvd__power_iteration_normalizer': ['auto', 'QR', 'LU', 'none'],
        "kneighborsclassifier__n_neighbors": [5, 10, 15, 20],
        'kneighborsclassifier__n_jobs': [-1]
    }
    KN_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    KN_cv.fit(X1, y1)
    params.append(KN_cv.best_params_)
    scores.append(KN_cv.best_score_)
    models.append(KN_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), LogisticRegression())
    param_grid = {
        "truncatedsvd__n_components": [300, 500],
        'truncatedsvd__n_iter': [20],
        'truncatedsvd__random_state': [0],
        'logisticregression__random_state': [0],
        'logisticregression__max_iter': [500],
        'truncatedsvd__power_iteration_normalizer': ['auto'],
        'logisticregression__penalty': ['l2', 'none'],
        'logisticregression__n_jobs': [-1]
    }
    LR_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    LR_cv.fit(X1, y1)
    params.append(LR_cv.best_params_)
    scores.append(LR_cv.best_score_)
    models.append(LR_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), SVC())
    param_grid = {
        "truncatedsvd__n_components": [300, 500],
        'truncatedsvd__n_iter': [20],
        'truncatedsvd__random_state': [0],
        'truncatedsvd__power_iteration_normalizer': ['auto', 'QR', 'LU', 'none'],
        'svc__random_state': [0],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    SVC_cv.fit(X1, y1)
    params.append(SVC_cv.best_params_)
    scores.append(SVC_cv.best_score_)
    models.append(SVC_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), DecisionTreeClassifier())
    param_grid = {
        "truncatedsvd__n_components": [300, 500],
        'truncatedsvd__n_iter': [20],
        'truncatedsvd__random_state': [0],
        'truncatedsvd__power_iteration_normalizer': ['auto', 'QR', 'LU', 'none'],
        'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss'],
        'decisiontreeclassifier__random_state': [0]
    }
    DT_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    DT_cv.fit(X1, y1)
    params.append(DT_cv.best_params_)
    scores.append(DT_cv.best_score_)
    models.append(DT_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), RandomForestClassifier())
    param_grid = {
        "truncatedsvd__n_components": [300, 500],
        'truncatedsvd__n_iter': [20],
        'truncatedsvd__random_state': [0],
        'truncatedsvd__power_iteration_normalizer': ['auto', 'QR', 'LU', 'none'],
        'randomforestclassifier__criterion': ['gini', 'entropy', 'log_loss'],
        'randomforestclassifier__max_depth': [1, 3, 5, 7, 9],
        'randomforestclassifier__random_state': [0],
        'randomforestclassifier__n_jobs': [-1]
    }
    RF_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    RF_cv.fit(X1, y1)
    params.append(RF_cv.best_params_)
    scores.append(RF_cv.best_score_)
    models.append(RF_cv.best_estimator_)

    model = make_pipeline(TruncatedSVD(), GaussianNB())
    vec = TfidfVectorizer()
    X = vec.fit_transform(train_messages)
    df2 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    all_features = df2.shape[1]
    feat_labels = vec.get_feature_names_out()
    X_train, X_test, y_train, y_test = train_test_split(df2, df['type'], random_state=0, train_size=0.7)

    param_grid = {
        "truncatedsvd__n_components": [300, 500],
        'truncatedsvd__n_iter': [20],
        'truncatedsvd__random_state': [0],
        'truncatedsvd__power_iteration_normalizer': ['auto', 'QR', 'LU', 'none']
    }
    CNB_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    CNB_cv.fit(X_train, y_train)
    params.append(CNB_cv.best_params_)
    scores.append(CNB_cv.best_score_)
    models.append(CNB_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), CategoricalNB())
    param_grid = {
        "truncatedsvd__n_components": [250, 500],
        'truncatedsvd__n_iter': [20],
        'truncatedsvd__random_state': [0],
        'truncatedsvd__power_iteration_normalizer': ['auto', 'QR', 'LU', 'none'],
        'categoricalnb__alpha': [.1, .3, .5, .7, .9, 1]
    }
    CNB_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    CNB_cv.fit(X1, y1)
    params.append(CNB_cv.best_params_)
    scores.append(CNB_cv.best_score_)
    models.append(CNB_cv.best_estimator_)

    highest = max(scores)
    ind = scores.index(highest)
    types = ['KNeighborsClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier',
             'RandomForestClassifier', 'GaussianNB', 'CategoricalNB']
    print('Model Types: ')
    print(types)
    print('Best cross validation scores from each of the seven model types:')
    print(scores)
    print('Model type with best cross validation score: ' + types[ind])
    print('Highest cross validation score for ' + types[ind] + ' is: ' + str(highest))
    print(types[ind] + ' parameters used to get best results:')
    print(params[ind])
    best_model = models[ind]
    print('Test set accuracy for ' + types[ind] + ' is: ' + str(accuracy_score(y2, best_model.predict(X2))))


if __name__ == '__main__':
    main()
