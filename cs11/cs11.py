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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from urllib.parse import urlparse


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


def countVowels(url):
    num_vowels = 0
    for char in url:
        if char in "aeiouAEIOU":
            num_vowels = num_vowels + 1
    return num_vowels


def count_domain_len(url):
    return len(urlparse(url).netloc)


def count_specials(url):
    num = 0
    for char in url:
        if char in "[@_!#$%^&*()<>?/\|}{~:]":
            num = num + 1
    return num


def calc_ratio(url):
    digit_count = 0
    for char in url:
        if char.isnumeric():
            digit_count += 1
    return digit_count / len(url)


def yes_or_no(i):
    i = i.lower().strip()
    if i[0] == 'y':
        return True
    if i[0] == 'n':
        return False


# feature ranking
# random forest can be used to rank features
# sequential backwards selection
def main():
    print("Default number of rows used is 1000, enter 1000 below")
    rows = input("Input the number of rows to be used in subset of csv data: ")
    rows = int(rows)
    df = pd.read_csv('malicious_phish.csv', nrows=rows)
    df['url'] = df['url'].apply(clean)
    list_of_urls = df['url'].to_list()

    # Train_test_split for pipelines
    X1, X2, y1, y2 = train_test_split(list_of_urls, df['type'], random_state=0, train_size=0.7)

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

    model = make_pipeline(GaussianNB())
    vec = TfidfVectorizer()
    X = vec.fit_transform(list_of_urls)
    df2 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    all_features = df2.shape[1]
    feat_labels = vec.get_feature_names_out()
    X_train, X_test, y_train, y_test = train_test_split(df2, df['type'], random_state=0, train_size=0.7)

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

    # Get best features then run with best model from above
    rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    rfc.fit(X_train, y_train)

    yn_input = input("Display the Most Important Features? (Yes or No): ")
    print_features = yes_or_no(yn_input)

    for feat in (2 ** p for p in range(1, 12)):
        select = SelectKBest(score_func=chi2, k=feat)
        select.fit(X_train, y_train)

        feature_list = []

        for feature_list_index in select.get_support(indices=True):
            feature_list.append(feat_labels[feature_list_index])
        if print_features:
            if feat == 2:
                print('Total number of features: ' + str(all_features))
            print(str(feat) + ' Most Important Features: ')
            print(feature_list)
        X_selected_train = select.transform(X_train)
        X_selected_test = select.transform(X_test)

        # Using best model type
        model = make_pipeline(SVC())
        param_grid = {
            'svc__random_state': [0],
            'svc__kernel': ['linear']
        }
        SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        SVC_cv.fit(X_selected_train, y_train)

        selected_pred = SVC_cv.predict(X_selected_test)
        print('Test set accuracy for ' + str(feat) + ' best features: ' + str(accuracy_score(y_test, selected_pred)))

    # Add vowel counts then run with best model from above
    list_vowel_counts = []
    for x in list_of_urls:
        list_vowel_counts.append(countVowels(x))
    df2['vowel_counts'] = list_vowel_counts
    X_train, X_test, y_train, y_test = train_test_split(df2, df['type'], random_state=0, train_size=0.7)

    model = make_pipeline(SVC())
    param_grid = {
        'svc__random_state': [0],
        'svc__kernel': ['linear']
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    SVC_cv.fit(X_train, y_train)
    vowels_pred = SVC_cv.predict(X_test)
    print('Test set accuracy for model after vowel count feature is added: ' + str(accuracy_score(y_test, vowels_pred)))

    list_domain_len = []
    for x in list_of_urls:
        list_domain_len.append(count_domain_len(x))
    df2['domain_len'] = list_domain_len
    X_train, X_test, y_train, y_test = train_test_split(df2, df['type'], random_state=0, train_size=0.7)

    model = make_pipeline(SVC())
    param_grid = {
        'svc__random_state': [0],
        'svc__kernel': ['linear']
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    SVC_cv.fit(X_train, y_train)
    domain_pred = SVC_cv.predict(X_test)
    print(
        'Test set accuracy for model after domain length feature is added: ' + str(accuracy_score(y_test, domain_pred)))

    list_special_counts = []
    for x in list_of_urls:
        list_special_counts.append(count_specials(x))
    df2['special_counts'] = list_special_counts
    X_train, X_test, y_train, y_test = train_test_split(df2, df['type'], random_state=0, train_size=0.7)

    model = make_pipeline(SVC())
    param_grid = {
        'svc__random_state': [0],
        'svc__kernel': ['linear']
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    SVC_cv.fit(X_train, y_train)
    special_pred = SVC_cv.predict(X_test)
    print('Test set accuracy for model after special character count feature is added: ' + str(
        accuracy_score(y_test, special_pred)))

    list_ratios = []
    for x in list_of_urls:
        list_ratios.append(calc_ratio(x))
    df2['digit_ratio'] = list_ratios
    X_train, X_test, y_train, y_test = train_test_split(df2, df['type'], random_state=0, train_size=0.7)

    model = make_pipeline(SVC())
    param_grid = {
        'svc__random_state': [0],
        'svc__kernel': ['linear']
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    SVC_cv.fit(X_train, y_train)
    ratio_pred = SVC_cv.predict(X_test)
    print('Test set accuracy for model after digit_count/url_length feature is added: ' + str(
        accuracy_score(y_test, ratio_pred)))
    # Attempted various combinations of additional features on top of full tfidfvectorizer feature set using the best
    # model and all provided an accuracy of 0.8766666666666667 when data rows was 1000


if __name__ == '__main__':
    main()
