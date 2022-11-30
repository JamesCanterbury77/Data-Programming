from IPython.core.display_functions import display
from password_strength import PasswordStats
import pandas as pd


def strength(row):
    stats = PasswordStats(row[0])
    return stats.strength()


def classify(row):
    value = row[1]
    if value < 0.33:
        return 'Weak'
    if 0.33 < value < .66:
        return 'Medium'
    if value > .66:
        return 'Strong'


df = pd.read_csv('passwords.csv', names=['Passwords'])
df = df.drop_duplicates(subset='Passwords', keep="first")
df['Strength'] = df.apply(strength, axis=1)
df['Class'] = df.apply(classify, axis=1)
display(df)
print(df['Class'].value_counts())
