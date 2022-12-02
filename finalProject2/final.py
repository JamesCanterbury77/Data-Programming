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
    list_of_messages = df['message'].to_list()

    # Train_test_split for pipelines
    X1, X2, y1, y2 = train_test_split(list_of_messages, df['Coding:Level1'], random_state=0, train_size=0.7)

    params = []
    scores = []
    models = []

    model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier())
    param_grid = {
        "kneighborsclassifier__n_neighbors": createList(1, 20)
    }
    KN_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    KN_cv.fit(X1, y1)
    params.append(KN_cv.best_params_)
    scores.append(KN_cv.best_score_)
    models.append(KN_cv.best_estimator_)


    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    param_grid = {
        'logisticregression__random_state': [0],
        'logisticregression__penalty': ['l2', 'none']
    }
    LR_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    LR_cv.fit(X1, y1)
    params.append(LR_cv.best_params_)
    scores.append(LR_cv.best_score_)
    models.append(LR_cv.best_estimator_)


    model = make_pipeline(TfidfVectorizer(), SVC())
    param_grid = {
        'svc__random_state': [0],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    SVC_cv.fit(X1, y1)
    params.append(SVC_cv.best_params_)
    scores.append(SVC_cv.best_score_)
    models.append(SVC_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())
    param_grid = {
        'decisiontreeclassifier__criterion': ['gini', 'entropy'],
        'decisiontreeclassifier__random_state': [0]
    }
    DT_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    DT_cv.fit(X1, y1)
    params.append(DT_cv.best_params_)
    scores.append(DT_cv.best_score_)
    models.append(DT_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
    param_grid = {
        'randomforestclassifier__criterion': ['gini', 'entropy'],
        'randomforestclassifier__max_depth': createList(1, 10),
        'randomforestclassifier__random_state': [0]
    }
    RF_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    RF_cv.fit(X1, y1)
    params.append(RF_cv.best_params_)
    scores.append(RF_cv.best_score_)
    models.append(RF_cv.best_estimator_)

    model = make_pipeline(GaussianNB())
    vec = TfidfVectorizer()
    X = vec.fit_transform(list_of_messages)
    df2 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    all_features = df2.shape[1]
    feat_labels = vec.get_feature_names_out()
    X_train, X_test, y_train, y_test = train_test_split(df2, df['Coding:Level1'], random_state=0, train_size=0.7)

    param_grid = {

    }
    CNB_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    CNB_cv.fit(X_train, y_train)
    params.append(CNB_cv.best_params_)
    scores.append(CNB_cv.best_score_)
    models.append(CNB_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    param_grid = {
        'multinomialnb__alpha': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    }
    MNB_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    MNB_cv.fit(X1, y1)
    params.append(MNB_cv.best_params_)
    scores.append(MNB_cv.best_score_)
    models.append(MNB_cv.best_estimator_)

    highest = max(scores)
    ind = scores.index(highest)
    types = ['KNeighborsClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier',
             'RandomForestClassifier', 'GaussianNB', 'MultinomialNB']
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
