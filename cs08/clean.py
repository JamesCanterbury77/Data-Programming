import pandas as pd
import string
import re
from city_name_map import city_name_map


def clean_city(city):
    if isinstance(city, str):
        city = city.strip()
        city = re.sub(r'\s+', ' ', city)
        city = city.title()
        if ',' in city:
            city = city.split(',', 1)[0]
        if city.endswith('Nc'):
            city = city[:-3]
        if city in city_name_map.keys():
            city = city_name_map[city]
        return city


def clean_state(state):
    if isinstance(state, str):
        state = state.upper()
        if state == 'NO':
            state = 'NC'
        return state


def clean_zip_code(zip):
    if isinstance(zip, str):
        z = ''
        for x in range(len(zip)):
            if zip[x].isnumeric():
                z += zip[x]
            if len(z) == 5:
                return z
        return float('nan')


def main():
    # Load and process the 'basic_person.csv' file.
    df = pd.read_csv('basic_person.csv')
    # clean the data here.
    df['city'] = df['city'].apply(clean_city)
    # print(df['city'].value_counts())
    df['state'] = df['state'].apply(clean_state)
    # print(df['state'].value_counts())
    df['zip'] = df['zip'].apply(clean_zip_code)
    # print(df['zip'].value_counts())
    # Save the file 'cleaned.csv' without the implicit index
    df.to_csv('cleaned.csv', index=False)


if __name__ == '__main__':
    main()
