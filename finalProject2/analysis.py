import pandas as pd
from IPython.core.display_functions import display


def main():
    df = pd.read_csv('roatan_train.csv')
    dfvalid = pd.read_csv('roatan_valid.csv')
    df = pd.concat([df, dfvalid], axis=0)

    print('Column Names:')
    print(df.columns)
    print('Dataframe:')
    display(df)

    print('Classification types for Coding:Level1')
    print(df['Coding:Level1'].unique())

    print(df['Coding:Level1'].value_counts())

    print('Classification types for Coding:Level2:')
    print(df['Coding:Level2'].unique())


if __name__ == '__main__':
    main()
