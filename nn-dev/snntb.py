import os
import configparser
from snntoolbox.bin.run import main
import sys

NUM_STEPS_PER_SAMPLE = 25  # Number of timesteps to run each sample (DURATION)
BATCH_SIZE = 64             # Affects memory usage. 32 -> 10 GB
NUM_TEST_SAMPLES = 100      # Number of samples to evaluate or use for inference
CONVERT_MODEL = True      

to_print = 'snntb requires 3 args (2nd and 3rd args are conditional) following these specifications:\n' +\
    'arg1:\n' +\
    '\tb -> Boolean Vectors (Doc2Vec)\n' +\
    '\tt -> TF-IDF Vectors\n' +\
    '\tr -> Random Square Vector\n' +\
    'arg2 (if arg1 is not provided as \'r\'):\n' +\
    '\tf -> Flat Encoding\n' +\
    '\ts -> Spiral Encoding\n' +\
    'arg3: (if arg2 is provided as \'f\')\n' +\
    '\tc -> Convolution Network\n' +\
    '\tf -> Feed-Forward Network'
if sys.argv[1].lower() == 'b':
    ENCODING1 = 'BOOL'
elif sys.argv[1].lower() == 't':
    ENCODING1 = 'TF_IDF'
elif sys.argv[1].lower() == 'r':
    ENCODING1 = 'RANDOM'
else:
    print(to_print)
    sys.exit(1)
if sys.argv[2].lower() == 'f':
    ENCODING2 = 'FLAT'
elif sys.argv[2].lower() == 's':
    ENCODING2 = 'SPIRAL'
else:
    print(to_print)
    sys.exit(1)
ENCODING3 = 'CN'
if ENCODING2 == 'FLAT':
    if sys.argv[3].lower() == 'c':
        ENCODING3 = 'CN'
    elif sys.argv[3].lower() == 'f':
        ENCODING3 = 'FFN'
    else:
        print(to_print)
        sys.exit(1)

E1 = ENCODING1.replace('_', '-').lower()
MODEL = f'{ENCODING2.lower()}_{ENCODING3.lower()}_{E1}'
MODEL_FILENAME = f'models/{MODEL}'
MODEL_NAME = MODEL_FILENAME.strip('.h5')
CURR_DIR = os.path.abspath('.')
ANN_MODEL_PATH = os.path.join(CURR_DIR, 'models', MODEL_FILENAME)
WORKING_DIR = os.path.join(CURR_DIR, '')
DATASET_DIR = os.path.join(CURR_DIR, f'npz/{ENCODING2.lower()}_{ENCODING1.lower()}_npz')

# Generate Config file
config = configparser.ConfigParser()
config['paths'] = {
    'path_wd': WORKING_DIR,
    'dataset_path': DATASET_DIR,
    'filename_ann': MODEL_NAME,
    'runlabel': MODEL_NAME+'_'+str(NUM_STEPS_PER_SAMPLE)
}
config['tools'] = {
    'evaluate_ann': True,
    'parse': True,
    'normalize': False,
    'simulate': True
}
config['simulation'] = {
    'simulator': 'INI',
    'duration': NUM_STEPS_PER_SAMPLE,
    'num_to_test': NUM_TEST_SAMPLES,
    'batch_size': BATCH_SIZE,
    'keras_backend': 'tensorflow'
}
config['output'] = {
    'verbose': 0,
    'plot_vars': {
        'input_image',
        'spiketrains',
        'spikerates',
        'spikecounts',
        'operations',
        'normalization_activations',
        'activations',
        'correlation',
        'v_mem',
        'error_t'
    },
    'overwrite': True
}

config_filepath = os.path.join(WORKING_DIR, 'data/config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# Convert the model using SNNToolbox
if __name__ == '__main__':
    
    print('Starting snntb...')
    main(config_filepath)