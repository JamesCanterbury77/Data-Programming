import string
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns


# USING BEST MODEL


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

    X1, X2, y1, y2 = train_test_split(train_messages, df['Coding:Level1'], random_state=0, train_size=0.7)

    params = []
    scores = []
    models = []

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(), SVC())
    param_grid = {
        'truncatedsvd__n_components': [325],
        'truncatedsvd__n_iter': [25],
        'truncatedsvd__random_state': [0],
        'svc__random_state': [0],
        'svc__kernel': ['rbf'],
        'svc__C': [1],
        'svc__shrinking': [True]
    }
    SVC_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
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
    y_model = best_model.predict(X2)
    print('Test set accuracy for ' + types[ind] + ' is: ' + str(accuracy_score(y2, y_model)))

    mat = confusion_matrix(y2, y_model)
    mat_df = pd.DataFrame(mat, index=['Action', 'Community', 'Information'],
                          columns=['Action', 'Community', 'Information'])
    sns.heatmap(mat_df, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.show()


if __name__ == '__main__':
    main()
