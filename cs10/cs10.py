from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def clean(line):
    if 'https://' in line:
        line = line[8:]
    if 'http://' in line:
        line = line[7:]
    line = line.replace('/', ' ')
    line = line.replace('.', ' ')
    line = line.replace('-', ' ')
    # print(line)
    return line


def createList(x, y):
    return [item for item in range(x, y + 1)]


# feature ranking
# random forest can be used to rank features
# sequential backwards selection
def main():
    print("Default number of rows used is 1000, enter 1000 below")
    rows = input("Input the number of rows to be used in subset of csv data: ")
    rows = int(rows)
    df = pd.read_csv('malicious_phish.csv', nrows=rows)
    df['type'] = df['type'].replace({'phishing': 'malicious', 'defacement': 'malicious', 'malware': 'malicious'})
    df['url'] = df['url'].apply(clean)
    list_of_urls = df['url'].to_list()

    X1, X2, y1, y2 = train_test_split(list_of_urls, df['type'], random_state=0, train_size=0.7)

    # get the best features
    vec = TfidfVectorizer()
    X = vec.fit_transform(list_of_urls)
    df2 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    feat_labels = vec.get_feature_names_out()
    X_train, X_test, y_train, y_test = train_test_split(df2, df['type'], random_state=0, train_size=0.7)

    rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    rfc.fit(X_train, y_train)

    select = SelectKBest(score_func=chi2, k=20)
    select.fit(X_train, y_train)

    feature_list = []

    for feature_list_index in select.get_support(indices=True):
        feature_list.append(feat_labels[feature_list_index])
    print('20 Most Important Features: ')
    print(feature_list)

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
        'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss'],
        'decisiontreeclassifier__random_state': [0]
    }
    DT_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    DT_cv.fit(X1, y1)
    params.append(DT_cv.best_params_)
    scores.append(DT_cv.best_score_)
    models.append(DT_cv.best_estimator_)

    model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
    param_grid = {
        'randomforestclassifier__criterion': ['gini', 'entropy', 'log_loss'],
        'randomforestclassifier__max_depth': createList(1, 10),
        'randomforestclassifier__random_state': [0]
    }
    RF_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    RF_cv.fit(X1, y1)
    params.append(RF_cv.best_params_)
    scores.append(RF_cv.best_score_)
    models.append(RF_cv.best_estimator_)

    highest = max(scores)
    ind = scores.index(highest)
    types = ['KNeighborsClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier',
             'RandomForestClassifier']
    print('Model Types: ')
    print(types)
    print('Best cross validation scores from each of the five model types')
    print(scores)
    print('Model type with best cross validation score: ' + types[ind])
    print('Highest cross validation score for ' + types[ind] + ' is: ' + str(highest))
    print(types[ind] + ' parameters used to get best results:')
    print(params[ind])
    best_model = models[ind]
    print('Test set accuracy for ' + types[ind] + ' is: ' + str(accuracy_score(y2, best_model.predict(X2))))


if __name__ == '__main__':
    main()
