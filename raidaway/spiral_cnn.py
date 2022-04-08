import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import time

# Path variables
ENCODING_PATH_DIC = {
    'TF_IDF': 'encodings/tf-idf_SPIRAL_encoding.txt',
    'BOOL': 'encodings/bool_SPIRAL_encoding.txt'
}
MODEL_PATH_DICT = {
    'TF_IDF': 'models/cnn_tf-idf.h5',
    'BOOL': 'models/cnn_bool.h5'
}

def num(arg):
    if arg is None:
        return 0
    else:
        return arg

# Model variables
ENCODING = 'BOOL' # 'TF_IDF' or 'BOOL'
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
'''
TF_IDF_ENCODING = []
r = open(TF_IDF_ENCODING_PATH, 'r')
for line in r.readlines():
    TF_IDF_ENCODING.append(line.split('\t'))
r.close()
BOOL_ENCODING = []
r = open(BOOL_ENCODING_PATH, 'r')
for line in r.readlines():
    BOOL_ENCODING.append(line.split('\t'))
r.close()
'''

CWS = []
r = open(ENCODING_PATH)
for line in r.readlines():
    line = line.strip()
    if line != '':
        CWS.append(line)
# CWS = CWS[-50:]

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

def get_data(row, encoding):

    global CWS
    global DATA
    global ENCODING
    global LABELLING
    
    tf_idf_vals = json.loads(row['tf_idf'])
    vector = []
    if ENCODING == 'TF_IDF':
        for cw in CWS:
            vector.append(num(tf_idf_vals.get(cw)))
    else:
        for cw in CWS:
            vector.append(int(num(tf_idf_vals.get(cw)) != 0))
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
    df = pd.read_excel('data/tf-idf.xlsx', engine='openpyxl')
    df.apply(lambda row: get_data(row, ENCODING), axis=1)
    LABELLING = np.array(LABELLING, np.int_)
    DATA = np.array(DATA, np.float_)
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

    input_shape = x_train[0].shape

    print('Creating model...')
    with strategy.scope():

        # Create the network model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=input_shape)) # Input
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        model.add(tf.keras.layers.Dropout(.2))
        model.add(tf.keras.layers.Dense(units=128, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(units=2)) # Output
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

    print('Starting simple_ffn...')
    main()
    