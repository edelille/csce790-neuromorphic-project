import math
import pandas as pd
import random

TF_IDF_VAL_DIFF_PATH = 'data/tf_idf_val_diff.xlsx'
BOOL_VAL_DIFF_PATH = 'data/bool_val_diff.xlsx'
TF_IDF_SPIRAL_ENCODING = 'encodings/spiral_tf_idf_encoding.txt'
BOOL_SPIRAL_ENCODING = 'encodings/spiral_bool_encoding.txt'
TF_IDF_DG_ENCODING = 'encodings/dg_tf_idf_encoding.txt'
BOOL_DG_ENCODING = 'encodings/dg_bool_encoding.txt'
TF_IDF_FLAT_ENCODING = 'encodings/flat_tf_idf_encoding.txt'
BOOL_FLAT_ENCODING = 'encodings/flat_bool_encoding.txt'
RANDOM_FLAT_ENCODING = 'encodings/flat_random_encoding.txt'
RANDOM_SQR_ENCODING = 'encodings/random_sqr_encoding.txt'


def spiral_str(keys):
    
    spiral_string = []
    n = int(math.sqrt(len(keys)))
    for i in range(0, n):
        this_str = ''
        for j in range(0, n):
            x = min(min(i, j), min(n - 1 - i, n - 1 - j))
            if i <= j:
                this_str += keys[abs((n - 2 * x) * (n - 2 * x) - (i - x) - (j - x) -(n**2 + 1))-1]
            else:
                this_str += keys[abs((n - 2 * x - 2) * (n - 2 * x - 2) + (i - x) + (j - x) - (n**2 + 1))-1]
            this_str += '\t'
        spiral_string.append(this_str)
    spiral_string = '\n'.join(spiral_string)
    return spiral_string

def diag_grad(keys):

    dim = int(len(keys)**.5)
    dg = []
    row = [0] * dim
    index = 0
    for a in range(0, dim):
        dg.append(row.copy())
    for a in range(0, dim):
        x = a
        y = 0
        for b in range(0, a + 1):
            dg[x][y] = index
            x -= 1
            y += 1
            index += 1
    for a in range(0, dim-1):
        x = dim - 1
        y = a + 1
        for b in range(0, dim-a-1):
            dg[x][y] = index
            x -= 1
            y += 1
            index += 1
    for a in range(0, dim):
        for b in range(0, dim):
            dg[a][b] = keys[dg[a][b]]
    dg_string = []
    for row in dg:
        dg_string.append('\t'.join(row))
    dg_string = '\n'.join(dg_string)
    return dg_string

def avg(arglist):
    
    return sum(arglist)/len(arglist)

def std(arglist):

    mean = avg(arglist)
    var = sum([((x - mean) ** 2) for x in arglist]) / len(arglist)
    return var ** .5

def main():

    print('Loading data...')
    df_tf_idf = pd.read_excel(TF_IDF_VAL_DIFF_PATH, engine='openpyxl')
    df_bool = pd.read_excel(BOOL_VAL_DIFF_PATH, engine='openpyxl')
    
    # Obtain list of each variable
    cw_tf_idf_list = list(df_tf_idf['word'])
    cw_bool_list = list(df_bool['word'])
    tf_idf_diff = list(df_tf_idf['tf_idf_diff'])
    bool_diff = list(df_bool['bool_diff'])
    tf_idf_ratio = list(df_tf_idf['bool_ratio'])
    bool_ratio = list(df_bool['tf_idf_ratio'])

    print(f'TF-IDF Diff Mean: {avg(tf_idf_diff)}')
    print(f'TF-IDF Diff STD: {std(tf_idf_diff)}')
    print(f'TF-IDF Ratio Mean: {avg(tf_idf_ratio)}')
    print(f'TF-IDF Ratio STD: {std(tf_idf_ratio)}')
    print(f'BOOL Diff Mean: {avg(bool_diff)}')
    print(f'BOOL Diff STD: {std(bool_diff)}')
    print(f'BOOL Ratio Mean: {avg(bool_ratio)}')
    print(f'BOOL Ratio STD: {std(bool_ratio)}')

    tf_idf_score = []
    bool_score = []

    print('Calculating weighted scores...')
    tf_idf_weight = (avg(tf_idf_ratio)/avg(tf_idf_diff))
    bool_weight = (avg(bool_ratio)/avg(bool_diff))
    for a in range(0, len(cw_tf_idf_list)):
        tf_idf_score.append((tf_idf_diff[a]*tf_idf_weight)+ tf_idf_ratio[a])
    for a in range(0, len(cw_bool_list)):
        bool_score.append((bool_diff[a]*bool_weight)+ bool_ratio[a])

    zip_tf_idf = [x for x in sorted(zip(tf_idf_score, cw_tf_idf_list))]
    zip_bool = [x for x in sorted(zip(bool_score, cw_bool_list))]

    print('Creating flat encoding...')
    cw_tf_idf = []
    cw_bool = []
    for a in range(0, len(cw_tf_idf_list)):
        cw_tf_idf.append(zip_tf_idf[a][1])
    for a in range(0, len(cw_bool_list)):
        cw_bool.append(zip_bool[a][1])

    print('Creating spiral (square) encoding matrix...')
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

    print('Creating diagonal gradient (square) encoding matrix...')
    tf_idf_dg = diag_grad(cw_tf_idf)
    bool_dg = diag_grad(cw_bool)

    print('Creating random (square) encoding matrix...')
    cw_random = cw_tf_idf.copy()
    random.shuffle(cw_random)
    random_spiral = spiral_str(cw_random)

    print('Saving encodings...')
    w = open(TF_IDF_SPIRAL_ENCODING, 'w')
    w.write(tf_idf_spiral)
    w.close()
    w = open(BOOL_SPIRAL_ENCODING, 'w')
    w.write(bool_spiral)
    w.close()
    w = open(TF_IDF_DG_ENCODING, 'w')
    w.write(tf_idf_dg)
    w.close()
    w = open(BOOL_DG_ENCODING, 'w')
    w.write(bool_dg)
    w.close()
    w = open(TF_IDF_FLAT_ENCODING, 'w')
    for cw in cw_tf_idf:
        w.write(cw + '\n')
    w.close()
    w = open(BOOL_FLAT_ENCODING, 'w')
    for cw in cw_bool:
        w.write(cw + '\n')
    w.close()
    w = open(RANDOM_FLAT_ENCODING, 'w')
    for cw in cw_random:
        w.write(cw + '\n')
    w.close()
    w = open(RANDOM_SQR_ENCODING, 'w')
    w.write(random_spiral)
    w.close()

if __name__ == '__main__':

    print('Starting img_layout...')
    main()
