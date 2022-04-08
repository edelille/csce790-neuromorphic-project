import json
import math
import nltk
from nltk.corpus import stopwords #, wordnet
# from nltk.stem.wordnet import WordNetLemmatizer
import os
import pandas as pd
# import spacy
import time

TRIAL_RUN = False # Recommended to set as True if first time running
READ_NUM = 2 # Number of transcripts we read if TRIAL RUN
ERR_COUNT = 0 # Number of transcripts unable/failed to parse
# SG_SIZE = 4
VID_CAP_PATH = 'data/transcripts.xlsx'
OUT_DOCS_PATH = 'data/docs.xlsx'
OUT_FREQ_WORDS_PATH = 'data/freq_words.xlsx'
# OUT_FREQ_SG_PATH = 'data/freq_sg.xlsx'

# Used for progress
COUNT = 0
CHECK = 0
TOTAL_TIME = 0

ESC = {
    r'\\n': ' ',
    r'\'': '\'',
    r'\\u2019': '\'',
    r'\\"': ' ',
    r'\\': ' ',
    '}\'': '}',
    '\'{': '{'
}

# lemma = nltk.stem.wordnet.WordNetLemmatizer()
# nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'morphologizer'])

def peruse(row, freq_words_df, freq_sg_df, docs_length):

    global COUNT
    global CHECK
    global TOTAL_TIME
    global ERR_COUNT
    global READ_NUM
    global ESC
    
    startTime = time.time()

    try:
        # Parse the raw JSON dumped string list
        lines = row['transcript_obj'][1:-1]
        for key in ESC.keys():
            lines = lines.replace(key, ESC[key])
        lines = lines.split('},')
        lines[-1] = lines[-1][1:-1]
        corpus = ''
        for a in range(0, lines.__len__()):
            try:
                corpus += json.loads(lines[a] + '}')['text'] + ' '
            except:
                pass
        corpus = corpus.replace('\n', ' ').replace('.', '. ').replace('  ', ' ')
        word_tokens = nltk.word_tokenize(corpus.lower())

        # Get vid_count, total_count of words
        # word_counts = []
        word_counts = {}
        # tokenizer generates word_tokens for characters or single words that are stopwords
        for a in range(0, word_tokens.__len__()):
            if not word_tokens[a] in stopwords.words() and word_tokens[a].__len__() >= 4:
                # Realized there is a much more effecient way of lemmatizing using another script
                '''
                pos_tag = nltk.pos_tag([word_tokens[a]])[0][1][:2]
                if pos_tag == 'JJ' or pos_tag == 'NN' or pos_tag == 'VB' or pos_tag == 'RB':
                    # Technically spacy is better, but remember spacy is designed to pipeline
                    # word_tokens[a] = lemma.lemmatize(word_tokens[a]) # nltk is not good at lemmatizing verbs
                    word_tokens[a] = nlp(word_tokens[a])[0].lemma_
                '''
                pass
            else: # stopword or too short
                # Mark for deletion
                word_tokens[a] = '*****'
        # Remove elements marked to be deleted
        word_tokens = [token for token in word_tokens if token != '*****']
        for token in set(word_tokens):
            # word_counts.append([token, word_tokens.count(token)])
            word_counts[token] = word_tokens.count(token)
            if token in freq_words_df.index:
                freq_words_df.loc[[token], ['vid_count']] += 1
                freq_words_df.loc[[token], ['total_count']] += word_tokens.count(token)
            else:
                freq_words_df.loc[token] = ({
                    'vid_count': 1,
                    'total_count': word_tokens.count(token)
                })
        
        '''
        # Get vid_count, total_count of skip grams
        sgs = set()
        for a in range(0, word_tokens.__len__() - SG_SIZE + 1):
            sgs.update(word_tokens[a:a + SG_SIZE])
        sgs = list(sgs)
        '''
        
        row['word_counts'] = json.dumps(word_counts)
        row['transcript_obj'] = ''
        # row['transcript'] = corpus
        # row['skip_gram'] = sgs

    except Exception as e:
        print("ERROR FOUND", e)
        row['err'] = 1
        ERR_COUNT += 1

    if TRIAL_RUN:
        READ_NUM -= 1
    
    COUNT += 1
    if (COUNT % 10 == 0):
        TOTAL_TIME += (time.time() - startTime)
        CHECK += 1
        print(f'({COUNT}/{docs_length}) ETC: {((docs_length-COUNT)*TOTAL_TIME/(CHECK))} s')

    return row

'''
def calc_idf(row, docs_shape):

    row['idf'] = round(math.log((1+docs_shape)/(1 + row['vid_count']) + 1), 3)
    return row
'''

def get_data(trial):

    assert os.path.exists(VID_CAP_PATH), f'ERROR --> data file path ({VID_CAP_PATH}) not found.'

    if trial:
        print('Reading initial {} transcripts...'.format(READ_NUM))
        docs = pd.read_excel(VID_CAP_PATH, engine='openpyxl', nrows=READ_NUM)
    else:
        print('Reading ALL transcripts...')
        docs = pd.read_excel(VID_CAP_PATH, engine='openpyxl')
    print('Finished Reading')

    docs.set_index('vid_id')
    # docs['transcript'] = ''
    docs['word_counts'] = ''
    # docs['skip_gram'] = ''
    docs['err'] = ''

    # WE NEED TO LEMMATIZE AND CURATE THE WORDS TO SEE WHAT WE WANT TO COMPUTE A VECTOR WITH
    freq_words = pd.DataFrame()
    freq_words['word'] = ''
    freq_words.set_index('word', inplace=True)
    freq_words['vid_count'] = 0
    freq_words['total_count'] = 0
    # freq_words['idf'] = 0

    '''
    # Repeat the experiment for skip-gram some other day
    freq_sgs = pd.DataFrame()
    freq_sgs['skip_gram'] = ''
    freq_sgs.set_index('sg', inplace=True)
    freq_sgs['vid_count'] = 0
    freq_sgs['total_count'] = 0
    freq_sgs['idf'] = 0
    '''

    print('Analyzing corpora...')
    # Change to None to freq_sgs in the event that we decide to do context based analysis
    docs = docs.apply(lambda row: peruse(row, freq_words, None, docs.shape[0]), axis=1)
    docs = docs[~(docs['err'] == 1)]
    print(f'Failed to read {ERR_COUNT} transcipts...')

    # Get list of value counts to make sure class ratio is still good
    print(docs['class'].value_counts())

    '''
    print('Calculating the IDF of words...')
    freq_words = freq_words.apply(lambda row: calc_idf(row, docs.shape[0]), axis=1)
    # freq_sgs = freq_sgs.apply(lambda row: calc_idf(row, docs.shape[0]), axis=1)
    '''

    # Temporary?
    docs.drop('transcript_obj', axis=1, inplace=True)
    docs.drop('err', axis=1, inplace=True)

    return docs, freq_words

def main():

    docs, freq_words = get_data(TRIAL_RUN)

    print('Saving data...')
    docs.to_excel(OUT_DOCS_PATH, index=False)
    freq_words.to_excel(OUT_FREQ_WORDS_PATH)

if __name__ == '__main__':
    
    print('Starting transcript_peruser...')
    startTime = time.time()
    main()
    exec_time = time.time() - startTime
    