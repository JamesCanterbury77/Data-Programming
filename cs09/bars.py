import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display_functions import display


# Removes the year and dash
def clean(data):
    data['Month Sold'] = data['Month Sold'][3:]
    return data


# Get all the subcategories
def createlist(line):
    subcat = line['SubCategory']
    if subcat not in savedsubcats:
        savedsubcats.append(subcat)
    return line


savedsubcats = []
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df3 = pd.DataFrame()


def main():
    subcat = input('Keep subcategories that start with: ')
    df = pd.read_csv('food_cleaned.csv')
    df = df[(df.SubCategory.str.startswith(subcat))]
    df = df[(df['Month Sold'].str.contains('19'))]
    df = df.apply(clean, axis=1)
    df = df.apply(createlist, axis=1)
    for x in range(len(savedsubcats)):
        savedsubcat = savedsubcats[x]
        df2 = df[(df.SubCategory == savedsubcat)].copy()
        df2['Month Sold'] = pd.Categorical(df2['Month Sold'], categories=month, ordered=True)
        df2 = df2.sort_values(by='Month Sold')
        df2 = df2.groupby(['Month Sold'])
        df2 = df2.sum()
        # Concatenates the data frames of each subcategory together
        if x == 0:
            df3 = df2
        else:
            df3 = pd.concat([df3, df2], axis=1)

    df3 = df3.sort_values(by='Month Sold')

    # Normalizes the counts
    for (columnName, columnData) in df3.items():
        if columnName != 'Month Sold':
            columnSum = df3[columnName].sum()
            df3[columnName] = df3[columnName] / columnSum

    ax = df3.plot(kind='bar', subplots=True, legend=False)

    # Plot Formatting
    plt.subplots_adjust(hspace=0, wspace=0)
    for x in range(len(ax)):
        ax[x].set_title('')
        ax[x].spines['top'].set_visible(False)
        ax[x].spines['bottom'].set_visible(False)
        ax[x].spines['left'].set_visible(False)
        ax[x].spines['right'].set_visible(False)
        ax[x].get_yaxis().set_visible(False)
        ax[x].set_ylim(0, 1)

    border = plt.figure(1).add_subplot(111)
    border.get_xaxis().set_visible(False)
    border.set_facecolor('none')
    border.set_ylim(0, len(savedsubcats))

    plt.figure(1).legend(ax, labels=savedsubcats)
    plt.ylabel('Normalized Counts')
    plt.show()


if __name__ == '__main__':
    main()
