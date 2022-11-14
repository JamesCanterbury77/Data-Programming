from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD


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


def convert_list(lst, labels):
    ret = []
    for x in range(0, len(lst)):
        if lst[x] == 0:
            ret.append(labels[0])
        if lst[x] == 1:
            ret.append(labels[1])
    return ret


def main():
    print("Default number of rows used is 1000, enter 1000 below")
    rows = input("Input the number of rows to be used in subset of csv data: ")
    rows = int(rows)
    df = pd.read_csv('malicious_phish.csv', nrows=rows)
    df['type'] = df['type'].replace({'phishing': 'malicious', 'defacement': 'malicious', 'malware': 'malicious'})
    df['url'] = df['url'].apply(clean)
    list_of_urls = df['url'].to_list()

    # Train_test_split for pipelines
    X1, X2, y1, y2 = train_test_split(list_of_urls, df['type'], random_state=0, train_size=0.7)

    print('KMeans')

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=300, n_iter=20, random_state=0,
                                                          power_iteration_normalizer='auto'), KMeans(n_clusters=2,
                                                                                                     random_state=0))

    labels = ['benign', 'malicious']
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('Score with zero value cluster being assigned benign: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=300, n_iter=20, random_state=0,
                                                          power_iteration_normalizer='auto'), KMeans(n_clusters=2,
                                                                                                     random_state=0))
    labels = ['malicious', 'benign']
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('Score with zero value cluster being assigned malicious: ' + str(accuracy_score(y_pred_t, y2)))

    components_scores = []
    x_scores = []
    for x in range(50, 900, 50):
        # print(x)
        model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x, n_iter=20, random_state=0,
                                                              power_iteration_normalizer='auto'), KMeans(n_clusters=2,
                                                                                                         random_state=0))
        model.fit(X1)
        y_pred = model.predict(X2)
        y_pred_t = convert_list(y_pred, labels)
        components_scores.append(accuracy_score(y_pred_t, y2))
        x_scores.append(x)

    highest = max(components_scores)
    ind = components_scores.index(highest)
    print('Best components value for TruncatedSVD: ' + str(x_scores[ind]))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='auto'), KMeans(n_clusters=2,
                                                                                                     random_state=0))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('Model with best components value: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='QR'), KMeans(n_clusters=2,
                                                                                                   random_state=0))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('TruncatedSVD: power_iteration_normalizer = QR: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='LU'), KMeans(n_clusters=2,
                                                                                                   random_state=0))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('TruncatedSVD: power_iteration_normalizer = LU: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='none'),
                          KMeans(n_clusters=2, random_state=0))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('TruncatedSVD: power_iteration_normalizer = none: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='none'),
                          KMeans(n_clusters=2, random_state=0,
                                 algorithm='elkan'))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('KMeans: algorithm = elkan: ' + str(accuracy_score(y_pred_t, y2)))

    # MIXTURES

    print('Gaussian Mixture')
    labels = ['malicious', 'benign']

    components_scores = []
    x_scores = []
    for x in range(50, 900, 50):
        # print(x)
        model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x, n_iter=20, random_state=0,
                                                              power_iteration_normalizer='auto'),
                              GaussianMixture(n_components=2,
                                              random_state=0))
        model.fit(X1)
        y_pred = model.predict(X2)
        y_pred_t = convert_list(y_pred, labels)
        components_scores.append(accuracy_score(y_pred_t, y2))
        x_scores.append(x)

    highest = max(components_scores)
    ind = components_scores.index(highest)
    print('Best components value for TruncatedSVD: ' + str(x_scores[ind]))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='auto'),
                          GaussianMixture(n_components=2,
                                          random_state=0))

    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('Model with best components value: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='QR'),
                          GaussianMixture(n_components=2,
                                          random_state=0))

    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('TruncatedSVD: power_iteration_normalizer = QR: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='LU'),
                          GaussianMixture(n_components=2,
                                          random_state=0))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('TruncatedSVD: power_iteration_normalizer = LU: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='none'),
                          GaussianMixture(n_components=2,
                                          random_state=0))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('TruncatedSVD: power_iteration_normalizer = none: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='auto'),
                          GaussianMixture(n_components=2,
                                          random_state=0, covariance_type='tied'))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('GaussianMixture: covariance_type = tied: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='auto'),
                          GaussianMixture(n_components=2,
                                          random_state=0, covariance_type='diag'))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('GaussianMixture: covariance_type = diag: ' + str(accuracy_score(y_pred_t, y2)))

    model = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=x_scores[ind], n_iter=20, random_state=0,
                                                          power_iteration_normalizer='auto'),
                          GaussianMixture(n_components=2,
                                          random_state=0, covariance_type='spherical'))
    model.fit(X1)
    y_pred = model.predict(X2)
    y_pred_t = convert_list(y_pred, labels)
    print('GaussianMixture: covariance_type = spherical: ' + str(accuracy_score(y_pred_t, y2)))


if __name__ == '__main__':
    main()
