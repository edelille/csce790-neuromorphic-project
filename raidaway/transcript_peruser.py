import json
import math
import nltk
from nltk.corpus import stopwords #, wordnet
# from nltk.stem.wordnet import WordNetLemmatizer
import os
import pandas as pd
import spacy
import time

''' Run this command for the first time '''
# python -m spacy download en_core_web_sm
''' Only uncomment if this (stopwords nltk download) is the first time running '''

TRIAL_RUN = False # Recommended to set as True if first time running
READ_NUM = 3 # Number of transcripts we read if TRIAL RUN
ERR_COUNT = 0 # Number of transcripts unable/failed to parse
# SG_SIZE = 4
VID_CAP_PATH = 'data/vid_captions.xlsx'
OUT_DOCS_PATH = 'data/docs.xlsx'
OUT_FREQ_WORDS_PATH = 'data/freq_words.xlsx'
# OUT_FREQ_SG_PATH = 'data/freq_sg.xlsx'

# Used for progress
COUNT = 0

# lemma = nltk.stem.wordnet.WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

def peruse(row, freq_words_df):

    global COUNT
    global ERR_COUNT
    global READ_NUM
    
    COUNT += 1
    startTime = time.time()

    try:
        lines = row['json_str'][1:-1].split('},')
        corpus = ''
        for a in range(0, lines.__len__()-1):
            corpus += json.loads(lines[a] + '}')['text'] + ' '
        corpus += json.loads(lines[-1])['text']
        corpus = corpus.replace('\n', ' ').replace('.', '. ').replace('  ', ' ')
        word_tokens = nltk.word_tokenize(corpus.lower())

        # Get vid_count, total_count of words
        # word_counts = []
        word_counts = {}
        # tokenizer generates word_tokens for characters or single words that are stopwords
        for a in range(0, word_tokens.__len__()):
            if not word_tokens[a] in stopwords.words() and word_tokens[a].__len__() >= 4:
                pos_tag = nltk.pos_tag([word_tokens[a]])[:2]
                if pos_tag == 'JJ' or pos_tag == 'NN' or pos_tag == 'VB' or pos_tag == 'RB':
                    # Technically spacy is better, but spacy is more designed for pipelines
                    # word_tokens[a] = lemma.lemmatize(word_tokens[a]) # nltk is not good at lemmatizing verbs
                    word_tokens[a] = nlp(word_tokens[a])[0].lemma_
            else: # stopword or too short
                # Mark for deletion
                word_tokens[a] = '*****'
        # Remove elements marked to be deleted
        word_tokens = filter(lambda val: val !=  '*****', word_tokens) 
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
        
        # Get vid_count, total_count of skip grams
        '''
        sgs = set()
        for a in range(0, word_tokens.__len__() - SG_SIZE + 1):
            sgs.update(word_tokens[a:a + SG_SIZE])
        sgs = list(sgs)
        
        row['transcript'] = corpus
        row['word_counts'] = word_counts
        row['skip_gram'] = sgs
        '''

    except Exception as e:
        row['err'] = 1
        print("ERROR FOUND", e)
        ERR_COUNT += 1

    if TRIAL_RUN:
        READ_NUM -= 1
        print(READ_NUM)

    print(time.time() - startTime, COUNT)
    return row

def calc_idf(row, docs_shape):

    row['idf'] = round(math.log((1+docs_shape)/(1 + row['vid_count']) + 1), 3)
    return row


def get_data(trial):

    assert os.path.exists(VID_CAP_PATH), f'ERROR --> data file path ({VID_CAP_PATH}) not found.'

    if trial:
        print('Reading initial {} transcripts...'.format(READ_NUM))
        docs = pd.read_excel(VID_CAP_PATH, header=None, engine='openpyxl', usecols=[0,1], nrows=READ_NUM)
    else:
        print('Reading ALL transcripts...')
        docs = pd.read_excel(VID_CAP_PATH, header=None, engine='openpyxl', usecols=[0,1])
    print('Finished Reading')

    docs.rename(columns = {
            0:'vid_id', 
            1:'json_str'
        }, inplace = True
    )
    docs.set_index('vid_id')
    docs['transcript'] = ''
    docs['word_counts'] = ''
    # docs['skip_gram'] = ''
    docs['err'] = ''

    # WE NEED TO CURATE THE WORDS TO SEE WHAT WE WANT TO COMPUTE A VECTOR WITH
    freq_words = pd.DataFrame()
    freq_words['word'] = ''
    freq_words.set_index('word', inplace=True)
    freq_words['vid_count'] = 0
    freq_words['total_count'] = 0
    freq_words['idf'] = 0

    '''
    freq_sgs = pd.DataFrame()
    freq_sgs['skip_gram'] = ''
    freq_sgs.set_index('sg', inplace=True)
    freq_sgs['vid_count'] = 0
    freq_sgs['total_count'] = 0
    freq_sgs['idf'] = 0
    '''

    print('Converting to JSON and analyzing corpus...')
    # Change to None to freq_sgs in the event that we decide to do context based analysis
    docs = docs.apply(lambda row: peruse(row, freq_words, None), axis=1)
    print(f'Failed to read {ERR_COUNT} transcipts...')
    
    print('Calculating the IDF of words...')
    freq_words = freq_words.apply(lambda row: calc_idf(row, docs.shape[0]), axis=1)
    # freq_sgs = freq_sgs.apply(lambda row: calc_idf(row, docs.shape[0]), axis=1)

    return docs, freq_words

def main():

    docs, freq = get_data(TRIAL_RUN)
    docs.to_excel(OUT_DOCS_PATH, index=False)
    freq.to_excel(OUT_FREQ_WORDS_PATH, index=False)

if __name__ == '__main__':
    
    print('Starting transcript_peruser...')

    startTime = time.time()
    main()
    exec_time = time.time() - startTime

    print(f'Execution time in seconds: {str(exec_time)})')
    