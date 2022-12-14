# from IPython.core.display_functions import display
# from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
import string
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD


def createList(x, y):
    return [item for item in range(x, y + 1)]


def clean(row):
    translator = str.maketrans('','', string.punctuation)
    row = row.translate(translator)
    return row


def main():
    df = pd.read_csv('roatan_train.csv')
    dfvalid = pd.read_csv('roatan_valid.csv')
    df = pd.concat([df, dfvalid], axis=0)
    df['message'] = df['message'].apply(clean)
    train_messages = df['message'].to_list()

    # Train_test_split
    X1, X2, y1, y2 = train_test_split(train_messages, df['Coding:Level1'], random_state=0, train_size=0.7)

    params = []
    scores = []
    models = []

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), SVC())
    param_grid = {
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'truncatedsvd__n_components': [150, 200, 300, 325],
        'truncatedsvd__n_iter': [5, 10, 15, 20],
        'truncatedsvd__random_state': [0],
        'svc__random_state': [0],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svc__C': [.3, .5, .7, .9, 1],
        'svc__shrinking': [True, False]
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    SVC_cv.fit(X1, y1)
    params.append(SVC_cv.best_params_)
    scores.append(SVC_cv.best_score_)
    models.append(SVC_cv.best_estimator_)

    highest = max(scores)
    ind = scores.index(highest)
    types = ['SVC']
    print('Model Type: SVC ')
    print('Highest cross validation score for SVC is: ' + str(highest))
    print(types[ind] + ' parameters used to get best results:')
    print(params[ind])
    best_model = models[ind]
    print('Test set accuracy for ' + types[ind] + ' is: ' + str(accuracy_score(y2, best_model.predict(X2))))


if __name__ == '__main__':
    main()
