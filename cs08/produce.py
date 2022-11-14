import pandas as pd

items = [
    'Apples, Early Yellow Transparent', 'Apples, Gala', 'Apples, Gold Rush',
    'Apples, Red Rome Beauty', 'Apples, Spice', 'Basil, Fresh - Sweet Genovese (green)',
    'Beets, Without Greens', 'Collards', 'Garlic Scapes', 'Jerusalem Artichokes',
    'Lettuce, Head', 'Lettuce, Loose Leaf Green', 'Microgreens, Sunshine Mix',
    'Okra, Green', 'Peppers, Bell (Green)', 'Peppers, Jalapeno', 'Pumpkin, Seminole',
    'Rosemary, Fresh', 'Sweet Potatoes, Orange', 'Watermelon, Jubilee'
]


def myhash(user_name):
    import hashlib
    m = hashlib.sha256()
    m.update(bytes(user_name, 'utf-8'))
    return int(m.hexdigest()[:16], 16)


user_name = 'canterburyjr'
item = items[myhash(user_name) % len(items)]
print(f'{user_name} cleans subcategory {item}')


def clean(sub):
    unit = '2 oz container'
    if sub['SubCategory'] == item:
        if sub['Unit'] == unit:
            return sub
        else:
            if sub['Unit'] == '1 lb container':
                sub['Unit'] = unit
                sub['Units Sold'] = 8
            elif sub['Unit'] == '1 oz container':
                sub['Unit'] = unit
                sub['Units Sold'] = .5
    return sub


def main():
    # Load and process the 'food.csv' file.
    df = pd.read_csv('food.csv')
    df = df.apply(clean, axis=1)
    # Save the file 'cleaned_produce.csv' without the implicit index
    df.to_csv('cleaned_produce.csv', index=False)


if __name__ == '__main__':
    main()
