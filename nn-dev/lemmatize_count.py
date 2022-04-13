import json
import nltk
from nltk.corpus import stopwords #, wordnet
# from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import spacy
import time


TRIAL_RUN = False # Recommended to set as True if first time running
READ_NUM = 10 # Number of transcripts we read if TRIAL RUN
DOCS_PATH = 'data/docs.xlsx'
FREQ_WORDS_PATH = 'data/freq_words.xlsx'
L_DOCS_PATH = 'data/lemma_docs.xlsx'
L_FW_PATH = 'data/lemma_freq_words.xlsx'

# Used for progress
COUNT1 = 0
CHECK1 = 0
TOTAL_TIME1 = 0
ERR1 = 0
COUNT2 = 0
CHECK2 = 0
TOTAL_TIME2 = 0
ERR2 = 0

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'morphologizer'])

def num(arg):
    if arg is None:
        return 0
    else:
        return arg

def lemmatize_docs(row, docs_length):

    global COUNT1
    global CHECK1
    global TOTAL_TIME1
    global ERR1

    startTime = time.time()

    word_count = json.loads(row['word_counts'])
    new_wcs = {}
    keys = word_count.keys()
    for key in keys:
        try:
            pos_tag = nltk.pos_tag([key])[0][1][:2]
            lemmatized_key = key
            if pos_tag == 'JJ' or pos_tag == 'NN' or pos_tag == 'VB' or pos_tag == 'RB':
                lemmatized_key = nlp(key)[0].lemma_
            new_wcs[lemmatized_key] = num(new_wcs.get(lemmatized_key)) + word_count[key]
        except:
            ERR1 += 1
    row['word_counts'] = json.dumps(new_wcs)

    COUNT1 += 1
    if (COUNT1 % 10 == 0):
        TOTAL_TIME1 += (time.time() - startTime)
        CHECK1 += 1
        print(f'({COUNT1}/{docs_length}) ETC: {((docs_length-COUNT1)*TOTAL_TIME1/(CHECK1))} s')

    return row

def lemmatize_freq(row, freq_length):

    global COUNT2
    global CHECK2
    global TOTAL_TIME2
    global ERR2

    startTime = time.time()

    try:
        pos_tag = nltk.pos_tag([row['word']])[0][1][:2]
        if pos_tag == 'JJ' or pos_tag == 'NN' or pos_tag == 'VB' or pos_tag == 'RB':
            row['word'] = nlp(row['word'])[0].lemma_
    except:
        ERR2 += 1

    COUNT2 += 1
    if (COUNT2 % 1000 == 0):
        TOTAL_TIME2 += (time.time() - startTime)
        CHECK2 += 1
        print(f'({COUNT2}/{freq_length}) ETC: {((freq_length-COUNT2)*TOTAL_TIME2/(CHECK2))} s')

    return row

def main(trial):

    if trial:
        docs = pd.read_excel(DOCS_PATH, engine='openpyxl', nrows=READ_NUM)
        freq_words = pd.read_excel(FREQ_WORDS_PATH, engine='openpyxl', nrows=READ_NUM*100)
    else:
        docs = pd.read_excel(DOCS_PATH, engine='openpyxl')
        freq_words = pd.read_excel(FREQ_WORDS_PATH, engine='openpyxl')

    print('Lemmatizing docs...')
    docs = docs.apply(lambda row: lemmatize_docs(row, docs.shape[0]), axis=1)
    print(f'Failed attempt {ERR1}/{docs.shape[0]} lemmatize_docs(...)')

    old_freq_len = freq_words.shape[0]

    print('Lemmatizing freq_words...')
    freq_words = freq_words.apply(lambda row: lemmatize_freq(row, freq_words.shape[0]), axis=1)
    print(f'Failed attempt {ERR2}/{old_freq_len} lemmatize_docs(...)')

    freq_words = freq_words.groupby(['word'], as_index=False).agg({'vid_count': 'sum', 'total_count': 'sum'})
    print(f'freq_word.shape[0]: {old_freq_len} -> {freq_words.shape[0]}')

    print('Saving data...')
    docs.to_excel(L_DOCS_PATH, index=False)
    freq_words.to_excel(L_FW_PATH, index=False)

if __name__ == '__main__':

    print('Starting lemmatize_count...')
    main(TRIAL_RUN)
    