import pandas as pd


def main():
    # Load and process the files.
    df_person = pd.read_csv('basic_person.csv')
    df_student = pd.read_csv('person_detail_f.csv')
    df_map = pd.read_csv('student_detail_v.csv')
    df_map = df_map.groupby(['student_id_new']).max('acct_id_new')
    df_map = df_map.groupby(['student_id_new']).max('person_detail_id_new')
    df_map = df_map.reset_index()
    df_map = df_map.merge(df_student, on='person_detail_id_new')
    df_map = df_map.merge(df_person, on='acct_id_new')
    # print(df_map)
    # Save the file 'joined.csv' without the implicit index
    df_map.to_csv('joined.csv', index=False)


if __name__ == '__main__':
    main()
