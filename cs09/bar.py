import matplotlib.pyplot as plt
import pandas as pd


# Removes the year and dash
def clean(data):
    data['Month Sold'] = data['Month Sold'][3:]
    return data


def main():
    subcat = input('Enter SubCategory: ')
    df = pd.read_csv('food_cleaned.csv')
    df = df[(df.SubCategory == subcat)]
    df = df[(df['Month Sold'].str.contains('19'))]
    df = df.apply(clean, axis=1)
    unit = df['Unit'].values[0].capitalize()
    month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Month Sold'] = pd.Categorical(df['Month Sold'], categories=month, ordered=True)
    df = df.sort_values(by='Month Sold')
    df = df.groupby('Month Sold', as_index=False)
    df = df.sum()
    df = df.query('`Units Sold` > 0')
    title_str = subcat + ' in ' + unit
    df.plot(x='Month Sold', y='Units Sold', kind='bar', title=title_str)
    plt.ylabel('Units Sold')
    plt.show()


if __name__ == '__main__':
    main()
