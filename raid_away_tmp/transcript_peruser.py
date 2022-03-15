import pandas as pd
import json
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import os
import spacy
import time

''' Run this command for the first time '''
# python -m spacy download en_core_web_sm
''' Only uncomment if this (stopwords nltk download) is the first time running '''
# nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

TRIAL_RUN = True # Recommended to set as True if first time running
READ_NUM = 1 # Number of transcripts we read
ERR_COUNT = 0 # Number of transcripts unable to parse
VID_CAP_PATH = 'vid_captions_abridged.xlsx'
OUT_DOCS_PATH = 'docs.csv'
OUT_FREQ_PATH = 'freq.csv'

# lemma = nltk.stem.wordnet.WordNetLemmatizer()

def convert(row, freq_df):

    global ERR_COUNT
    global READ_NUM
    ''' Maybe add limit on how short/long the video can be '''
    # print(len(row['json_str']))
    lines = row['json_str'][1:-1].split('},')
    corpus = ''
    words_counts = []
    try:
        for a in range(0, lines.__len__()-1):
            corpus += json.loads(lines[a] + '}')['text'] + ' '
        corpus += json.loads(lines[-1])['text']
        corpus = corpus.replace('\n', ' ').replace('.', '. ').replace('  ', ' ')
        word_tokens = nltk.word_tokenize(corpus.lower())
        # tokenizer generates word_tokens for characters or single words that are stopwords
        for token in word_tokens:
            if not token in stopwords.words() and token.__len__() > 1:
                if token.__len__() >= 5:
                    pos_tag = nltk.pos_tag([token])[:2]
                    print
                    if pos_tag == 'JJ' or pos_tag == 'NN' or pos_tag == 'VB':
                        # token = lemma.lemmatize(token) # nltk is not good at lemmatizing verbs
                        token = nlp(token)[0].lemma_ # Technically spacy is better, but spacy is more designed for pipelines
        for token in set(word_tokens):
            words_counts.append([token, word_tokens.count(token)])
            if token in freq_df.index:
                freq_df.loc[[token], ['vid_count']] += 1
                freq_df.loc[[token], ['total_count']] += word_tokens.count(token)
            else:
                freq_df.loc[token] = ({
                    'vid_count': 1,
                    'total_count': word_tokens.count(token)
                })
        row['transcript'] = corpus
        row['words_counts'] = words_counts
    except:
        ERR_COUNT += 1
    READ_NUM -= 1
    print(READ_NUM)
    return row

def get_data(trial):

    assert os.path.exists(VID_CAP_PATH), f'ERROR --> data file path ({VID_CAP_PATH}) not found.'

    if trial:
        print('Reading initial {} transcripts'.format(READ_NUM))
        docs = pd.read_excel(VID_CAP_PATH, header=None, engine='openpyxl', usecols=[0,1], nrows=READ_NUM)
    else:
        print('Reading all transcripts')
        docs = pd.read_excel(VID_CAP_PATH, header=None, engine='openpyxl', usecols=[0,1])
    print('Finished Reading')

    docs.rename(columns = {
            0:'vid_id', 
            1:'json_str'
        }, inplace = True
    )
    docs.set_index('vid_id')
    docs['transcript'] = ''
    docs['words_counts'] = ''
    docs['occurence_map'] = ''

    # WE NEED TO CURATE THE WORDS TO SEE WHAT WE WANT TO COMPUTE A VECTOR WITH
    freq = pd.DataFrame()
    freq['word'] = ''
    freq = freq.set_index('word')
    freq['vid_count'] = 0
    freq['total_count'] = 0

    print('Converting to JSON and analyzing')
    docs = docs.apply(lambda row: convert(row, freq), axis=1)
    print(f'Failed to read {ERR_COUNT} transcipts...')

    # Remove all rare words
    freq = curate(freq)
    freq.to_csv('freq.csv')

    return docs, freq

def curate(freq):

    #####
    freq = freq[freq.vid_count != 1]
    return freq[freq.total_count != 1]

def main():

    docs, freq = get_data(TRIAL_RUN)
    docs.to_csv(OUT_DOCS_PATH)
    freq.to_csv(OUT_FREQ_PATH)
    curate(freq)

if __name__ == '__main__':
    print('Starting transcript_peruser...')

    startTime = time.time()

    main()

    exec_time = time.time() - startTime
    print(f'Execution time in seconds: {str(exec_time)})')