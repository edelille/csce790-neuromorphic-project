import json
import pandas as pd
# import time

CW_PATH = 'data/curated_words.xlsx'
TF_IDF_PATH = 'data/tf_idf.xlsx'
OUT_TF_IDF_PATH = 'data/tf_idf_val_diff.xlsx'
OUT_BOOL_PATH = 'data/bool_val_diff.xlsx'

def num(arg):
    if arg is None:
        return 0
    else:
        return arg

def main():
    
    print('Loading data...')
    cw = pd.read_excel(CW_PATH, engine='openpyxl')
    cw_list = list(cw['word'])
    df = pd.read_excel(TF_IDF_PATH, engine='openpyxl')
    df0 = df[(df['class'] == 'no')]
    df1 = df[(df['class'] == 'yes')]
    del df

    data = {}
    for cw in cw_list:
        data[cw] = {
            'count_0': 0,
            'count_1': 0,
            'tf_idf_0': 0,
            'tf_idf_1': 0,
            'tf_idf_diff': 0,
            'bool_0': 0,
            'bool_1': 0,
            'bool_diff': 0
        }
    
    print('Getting values for class 0...')
    for a in range(0, df0.shape[0]):
        tf_idf_vals = json.loads(df0.iloc[a]['tf_idf'])
        for cw in cw_list:
            data[cw]['count_0'] += 1
            data[cw]['tf_idf_0'] += num(tf_idf_vals.get(cw))
            data[cw]['bool_0'] += 0 if num(tf_idf_vals.get(cw)) == 0 else 1
    
    print('Getting values for class 1...')
    for a in range(0, df1.shape[0]):
        tf_idf_vals = json.loads(df1.iloc[a]['tf_idf'])
        for cw in cw_list:
            data[cw]['count_1'] += 1
            data[cw]['tf_idf_1'] += num(tf_idf_vals.get(cw))
            data[cw]['bool_1'] += 0 if num(tf_idf_vals.get(cw)) == 0 else 1

    df = pd.DataFrame()
    df['word'] = ''
    df['tf_idf_0'] = 0
    df['tf_idf_1'] = 0
    df['tf_idf_diff'] = 0
    df['tf_idf_ratio'] = 0
    df['bool_0'] = 0
    df['bool_1'] = 0
    df['bool_diff'] = 0
    df['bool_ratio'] = 0

    print('Creating dataframe...')
    for cw in cw_list:
        data[cw]['tf_idf_0'] = round(data[cw]['tf_idf_0']/data[cw]['count_0'], 5)
        data[cw]['tf_idf_1'] = round(data[cw]['tf_idf_1']/data[cw]['count_1'], 5)
        data[cw]['bool_0'] = round(data[cw]['bool_0']/data[cw]['count_0'], 5)
        data[cw]['bool_1'] = round(data[cw]['bool_1']/data[cw]['count_1'], 5)
        data[cw]['tf_idf_diff'] = round(data[cw]['tf_idf_1'] - data[cw]['tf_idf_0'], 5)
        data[cw]['bool_diff'] = round(data[cw]['bool_1'] - data[cw]['bool_0'], 5)
        try:
            data[cw]['tf_idf_ratio'] = max(min(round(data[cw]['tf_idf_1']/data[cw]['tf_idf_0'], 5), 10), 0)
            data[cw]['bool_ratio'] = max(min(round(data[cw]['bool_1']/data[cw]['bool_0'], 5), 10), 0)
        except:
            data[cw]['tf_idf_ratio'] = 10
            data[cw]['bool_ratio'] = 10
        new_row = pd.DataFrame({
            'word': cw,
            'tf_idf_0': data[cw]['tf_idf_0'],
            'tf_idf_1': data[cw]['tf_idf_1'],
            'tf_idf_diff': data[cw]['tf_idf_diff'],
            'tf_idf_ratio': data[cw]['tf_idf_ratio'],
            'bool_0': data[cw]['bool_0'],
            'bool_1': data[cw]['bool_1'],
            'bool_diff': data[cw]['bool_diff'],
            'bool_ratio': data[cw]['bool_ratio']
        }, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)
        del new_row
    
    print('Applying masks...')
    df_tf_idf = df[(df['tf_idf_diff'] >= 1) | (df['tf_idf_ratio'] >= 2)]
    print(f'TF_IDF_VAL_DIFF: {df.shape[0]} -> {df_tf_idf.shape[0]}')
    df_bool = df[(df['bool_diff'] >= .1) | (df['bool_ratio'] >= 2)]
    print(f'BOOL_VAL_DIFF: {df.shape[0]} -> {df_bool.shape[0]}')
    
    print('Saving data...')
    df_tf_idf.to_excel(OUT_TF_IDF_PATH, index=False)
    df_bool.to_excel(OUT_BOOL_PATH, index=False)

if __name__ == '__main__':

    print('Starting vals_diff.py')
    main()
    