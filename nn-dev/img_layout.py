import math
import pandas as pd

TF_IDF_VAL_DIFF_PATH = 'data/tf-idf_val_diff.xlsx'
BOOL_VAL_DIFF_PATH = 'data/bool_val_diff.xlsx'
TF_IDF_SPIRAL_ENCODING = 'encodings/tf-idf_spiral_encoding.txt'
BOOL_SPIRAL_ENCODING = 'encodings/bool_spiral_encoding.txt'
TF_IDF_FLAT_ENCODING = 'encodings/tf-idf_FLAT_encoding.txt'
BOOL_FLAT_ENCODING = 'encodings/bool_FLAT_encoding.txt'

def spiral_str(keys):
    
    spiral_string = ''
    n = int(math.sqrt(len(keys)))
    for i in range(0, n):
        for j in range(0, n):
            x = min(min(i, j), min(n - 1 - i, n - 1 - j))
            if i <= j:
                spiral_string += keys[abs((n - 2 * x) * (n - 2 * x) - (i - x) - (j - x) -(n**2 + 1))-1]
            else:
                spiral_string += keys[abs((n - 2 * x - 2) * (n - 2 * x - 2) + (i - x) + (j - x) - (n**2 + 1))-1]
            spiral_string += '\t'
        spiral_string += '\n'
    return spiral_string

def main():

    print('Loading data...')
    df_tf_idf = pd.read_excel(TF_IDF_VAL_DIFF_PATH, engine='openpyxl')
    df_bool = pd.read_excel(BOOL_VAL_DIFF_PATH, engine='openpyxl')
    
    # Obtain list of each variable
    cw_tf_idf_list = list(df_tf_idf['word'])
    cw_bool_list = list(df_bool['word'])
    tf_idf_diff = list(df_tf_idf['tf_idf_diff'])
    bool_diff = list(df_bool['bool_diff'])

    zip_tf_idf = [x for x in sorted(zip(tf_idf_diff, cw_tf_idf_list))]
    zip_bool = [x for x in sorted(zip(bool_diff, cw_bool_list))]

    print('Creating flat encoding...')
    cw_tf_idf = []
    cw_bool = []
    for a in range(0, len(cw_tf_idf_list)):
        cw_tf_idf.append(zip_tf_idf[a][1])
    for a in range(0, len(cw_bool_list)):
        cw_bool.append(zip_bool[a][1])

    print('Creating spiral encoding matrix...')
    sqrt = math.floor(math.sqrt(df_tf_idf.shape[0]))
    if sqrt % 2 == 1:
        n = df_tf_idf.shape[0] - sqrt**2
        cw_tf_idf = cw_tf_idf[n:]
    else:
        n = (sqrt+1)**2 - df_tf_idf.shape[0]
        sqrt += 1
        cw_tf_idf = cw_tf_idf + cw_tf_idf[-n:]
    sqrt = math.floor(math.sqrt(df_bool.shape[0]))
    if sqrt % 2 == 1:
        n = df_bool.shape[0] - sqrt**2
        cw_bool = cw_bool[n:]
    else:
        n = (sqrt+1)**2 - df_bool.shape[0]
        sqrt += 1
        cw_bool = cw_bool + cw_bool[-n:]
    tf_idf_spiral = spiral_str(cw_tf_idf)
    bool_spiral = spiral_str(cw_bool)

    print('Saving encodings...')
    w = open(TF_IDF_SPIRAL_ENCODING, 'w')
    w.write(tf_idf_spiral)
    w.close()
    w = open(BOOL_SPIRAL_ENCODING, 'w')
    w.write(bool_spiral)
    w.close()
    w = open(TF_IDF_FLAT_ENCODING, 'w')
    for cw in cw_tf_idf:
        w.write(cw + '\n')
    w.close()
    w = open(BOOL_FLAT_ENCODING, 'w')
    for cw in cw_bool:
        w.write(cw + '\n')
    w.close()

if __name__ == '__main__':

    print('Starting img_layout...')
    main()