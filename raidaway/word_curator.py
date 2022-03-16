# from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import spacy

INPUT_PATH = 'data/freq.csv'
OUTPUT_PATH = 'data/curated_freq.csv'
OUTPUT_REMOVED_PATH = 'data/removed_freq.csv'

# Tune according to full data set
VID_COUNT_FLOOR = 15
TOTAL_COUNT_FLOOR = 15

print('Loading data...')
freq = pd.read_csv(INPUT_PATH)

# lemma = nltk.stem.wordnet.WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

removed = pd.DataFrame()
removed['word'] = ''
removed['vid_count'] = 0
removed['total_count'] = 0

# Allows us to apply masks while also letting us maintain record of words removed
def both_mask(mask):

    global freq
    global removed

    return freq[mask], removed.append(freq[~mask])

def lemma_word(row):

    # new_word = lemma.lemmatize(row.word)
    new_word = nlp(row.word)[0].lemma_
    '''
    if row.word != new_word: 
        print('{} -> {}'.format(row.word, new_word))
        row.word = new_word
    '''
    return row

def main():

    global freq
    global removed

    freq = freq.astype({'word':'string'})
    freq.dropna(inplace=True)

    orig_count = freq.shape[0]

    # Realized this is already accomplished in transcript_peruser.py
    '''
    print('Lemmatizing and Aggregating rows by \'word\'...')
    freq = freq.apply(lambda row: lemma_word(row), axis=1)
    agg_funcs = {'vid_count': 'sum', 'total_count': 'sum'}
    freq = freq.groupby(freq.word, as_index=False).aggregate(agg_funcs)

    # print('Row count after curating possible duped words: {} -> {}'.format(orig_count, freq.shape[0]))

    orig_count = freq.shape[0]
    '''

    print('Applying masks...')
    freq, removed = both_mask((freq.vid_count >= VID_COUNT_FLOOR))
    freq, removed = both_mask((freq.total_count >= TOTAL_COUNT_FLOOR))
    freq, removed = both_mask((freq.word.str.len() >= 3))
    blacklist = '1234567890'
    for char in blacklist:
        freq, removed = both_mask(~(freq.word.str.contains(char)))
        
    print('Row count after curating with mask: {} -> {}'.format(orig_count, freq.shape[0]))

    print('Saving data...')
    freq.to_csv(OUTPUT_PATH, index=False)
    removed.to_csv(OUTPUT_REMOVED_PATH, index=False)

if __name__ == '__main__':

    main()