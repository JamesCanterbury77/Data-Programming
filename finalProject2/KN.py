from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
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

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), KNeighborsClassifier())
    param_grid = {
        "truncatedsvd__n_components": [25, 50, 100, 200, 300],
        'truncatedsvd__n_iter': [20, 25, 30, 35],
        'truncatedsvd__random_state': [0],
        "kneighborsclassifier__n_neighbors": [5, 10, 15, 20],
        'kneighborsclassifier__n_jobs': [-1]
    }
    KN_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    KN_cv.fit(X1, y1)
    params.append(KN_cv.best_params_)
    scores.append(KN_cv.best_score_)
    models.append(KN_cv.best_estimator_)

    highest = max(scores)
    ind = scores.index(highest)
    types = ['KN']
    print('Model Type: KN ')
    print('Highest cross validation score for KN is: ' + str(highest))
    print(types[ind] + ' parameters used to get best results:')
    print(params[ind])
    best_model = models[ind]
    print('Test set accuracy for ' + types[ind] + ' is: ' + str(accuracy_score(y2, best_model.predict(X2))))


if __name__ == '__main__':
    main()
