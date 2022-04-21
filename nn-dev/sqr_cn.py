import json
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import time

def num(arg):
    if arg is None:
        return 0
    else:
        return arg

to_print = 'sqr_cn only accepts 1 of the following for its single argument:\n' +\
        'spb -> Spiral Boolean Vector (BagOfBooleans)\n' +\
        'spt -> Sprial TF-IDF Square Vector\n' +\
        'dgb -> Diagonal Gradient Boolean Vector (BagOfBooleans)\n' +\
        'dgt -> Diagonal TF-IDF Vector\n' +\
        'r -> Random Square Vector\n'
if len(sys.argv) != 2:
    print(to_print)
    sys.exit(1)
if sys.argv[1].lower() == 'spb':
    PATTERN = 'SPIRAL'
    ENCODING = 'BOOL'
elif sys.argv[1].lower() == 'spt':
    PATTERN = 'SPIRAL'
    ENCODING = 'TF_IDF'
elif sys.argv[1].lower() == 'dgb':
    PATTERN = 'DG'
    ENCODING = 'BOOL'
elif sys.argv[1].lower() == 'dgt':
    PATTERN = 'DG'
    ENCODING = 'TF_IDF'
elif sys.argv[1].lower() == 'r':
    ENCODING = 'RANDOM'
else:
    print(to_print)
    sys.exit(1)

DATA_PATH = 'data/tf_idf.xlsx'
ENCODING_PATH_DIC = {
    'BOOL': f'encodings/{PATTERN.lower()}_bool_encoding.txt',
    'TF_IDF': f'encodings/{PATTERN.lower()}_tf_idf_encoding.txt',
    'RANDOM': 'encodings/random_sqr_encoding.txt'
}
MODEL_PATH_DICT = {
    'BOOL': f'models/{PATTERN.lower()}_bool_cn.h5',
    'TF_IDF': f'models/{PATTERN.lower()}_tf_idf_cn.h5',
    'RANDOM': 'models/random_sqr_cn.h5'
}
if ENCODING != 'RANDOM':
    SAVE_DATA_DIR_PATH = f'npz/{PATTERN.lower()}_{ENCODING.lower()}_npz/'
else:
    SAVE_DATA_DIR_PATH = f'npz/random_sqr_npz/'

TRAIN_PERC = 50
TEST_PERC = 25
VAL_PERC = 25
EPOCHS = 20
BATCH_SIZE = 100
DATA = []
LABELLING = []

ENCODING_PATH = ENCODING_PATH_DIC[ENCODING]
MODEL_OUT_PATH = MODEL_PATH_DICT[ENCODING]

# Default usage of CPU
device_type = 'CPU'
devices = tf.config.experimental.list_physical_devices(device_type)
device_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.OneDeviceStrategy(device_names[0])

# Load encoding matrix (Don't use in simple ffn)
ENCODING = []
r = open(ENCODING_PATH, 'r')
for line in r.readlines():
    ENCODING.append(line.split('\t'))
r.close()

CWS = []
r = open(ENCODING_PATH)
for line in r.readlines():
    line = line.strip()
    if line != '':
        line = line.split('\t')
        CWS.append(line)

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_OUT_PATH,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=0,
    mode='min'
)
callbacks = [checkpoint]

def get_data(row):

    global CWS
    global DATA
    global ENCODING
    global LABELLING
    
    tf_idf_vals = json.loads(row['tf_idf'])
    vector = []
    for cw_list in CWS:
        vector_row = []
        for cw in cw_list:
            if ENCODING == 'TF_IDF':
                vector_row.append(num(tf_idf_vals.get(cw)))
            else: # ENCODING == 'BOOL':
                vector_row.append(int(num(tf_idf_vals.get(cw)) != 0))
        vector.append(vector_row)
    DATA.append(vector)
    if row['class'] == 'no': # Class 0
        LABELLING.append([1, 0])
    else: # row['class'] == 'yes': # Class 1
        LABELLING.append([0, 1])

def split(data):

    global TEST_PERC
    global VAL_PERC

    train, test = train_test_split(data, test_size=(TEST_PERC + VAL_PERC)/100)
    test, val = train_test_split(test, test_size=VAL_PERC/(TEST_PERC + VAL_PERC))
    return train, test, val

def main():

    global DATA
    global ENCODING
    global LABELLING

    print('Getting, converting, and splitting data...')
    df = pd.read_excel(DATA_PATH, engine='openpyxl')
    df.apply(lambda row: get_data(row), axis=1)
    LABELLING = np.array(LABELLING, np.int_)
    DATA = np.array(DATA, np.float_)
    '''
        Another way of adding another dimension to the data
        Preferable since the reshape portion is left out of the NN scheme.
        Do note, this can take time
    '''
    DATA = tf.expand_dims(DATA, axis=-1)
    train, test, val = split(list(zip(DATA, LABELLING)))

    x_train = [x[0] for x in train]
    y_train = [x[1] for x in train]
    x_test = [x[0] for x in test]
    y_test = [x[1] for x in test]
    x_val = [x[0] for x in val]
    y_val = [x[1] for x in val]

    x_train = np.array(x_train, np.float_)
    y_train = np.array(y_train, np.int_)
    x_test = np.array(x_test, np.float_)
    y_test = np.array(y_test, np.int_)
    x_val = np.array(x_val, np.float_)
    y_val = np.array(y_val, np.int_)

    print('Saving data...')
    np.savez_compressed(f'{SAVE_DATA_DIR_PATH}x_train', x_train)
    np.savez_compressed(f'{SAVE_DATA_DIR_PATH}y_train', y_train)
    np.savez_compressed(f'{SAVE_DATA_DIR_PATH}x_test', x_test)
    np.savez_compressed(f'{SAVE_DATA_DIR_PATH}y_test', y_test)
    np.savez_compressed(f'{SAVE_DATA_DIR_PATH}x_val', x_val)
    np.savez_compressed(f'{SAVE_DATA_DIR_PATH}y_val', y_val)

    input_shape = x_train[0].shape

    print('Creating model...')
    with strategy.scope():

        # Create the network model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=input_shape)) # Input
        model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=2, strides=2))
        model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=2, strides=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dropout(.2))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=2, activation='sigmoid')) # Output
       
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    model.summary()

    print('Training model...')
    model.fit(
        x_train, y_train, 
        epochs=EPOCHS,
        verbose=2,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        batch_size=BATCH_SIZE
    )

    print('Evaluating model...')
    eval_start = time.time()
    loss, acc = model.evaluate(
        x_test,
        y_test,
        verbose=2
    )
    total_time = time.time() - eval_start
    print(f'Evaluation performed on {len(x_test)} samples batches of {BATCH_SIZE}')
    print(f'Latency: {total_time}')
    print(f'Loss: {loss}')
    print(f'Accuracy: {acc}')

if __name__ == '__main__':

    print('Starting sqr_cn...')
    main()
    