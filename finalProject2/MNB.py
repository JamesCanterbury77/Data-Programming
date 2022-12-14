from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
import string


def createList(x, y):
    return [item for item in range(x, y + 1)]


def clean(row):
    translator = str.maketrans('', '', string.punctuation)
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

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    param_grid = {
        'multinomialnb__alpha': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        'multinomialnb__fit_prior': [True, False]
    }
    MNB_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    MNB_cv.fit(X1, y1)
    params.append(MNB_cv.best_params_)
    scores.append(MNB_cv.best_score_)
    models.append(MNB_cv.best_estimator_)

    highest = max(scores)
    ind = scores.index(highest)
    types = ['MultinomialNB']
    print('Model Type: MultinomialNB ')
    print('Highest cross validation score for MultinomialNB is: ' + str(highest))
    print(types[ind] + ' parameters used to get best results:')
    print(params[ind])
    best_model = models[ind]
    print('Test set accuracy for ' + types[ind] + ' is: ' + str(accuracy_score(y2, best_model.predict(X2))))


if __name__ == '__main__':
    main()
