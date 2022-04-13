import pandas as pd

OUT_CLEAN = 'data/clean_data.xlsx'
DIRTY_PATH = 'data/dirty_data.xlsx'
OUT_SPONSORS = 'data/sponsor_words.txt'

def main():

    # Helps deal with invisible columns
    cols = ['Video ID','URL','Does it have an ad?','Sponsor name']
    df = pd.read_excel(DIRTY_PATH, engine='openpyxl', usecols=cols)

    # Keep only labelled data and clean as necessary
    df.rename(columns={'Video ID': 'vid_id', 'URL': 'url', 'Does it have an ad?': 'class'}, inplace=True)
    df.dropna(subset=['class'], inplace=True)
    df['class'] = df['class'].str.lower()
    df['class'] = df['class'].str.strip()
    df = df[(df['class'] == 'yes') | (df['class'] == 'no')]
    df.sort_values(by=['class'], inplace=True)

    #print(df['class'].value_counts())
    '''
    Terminal Output:

    no     454
    yes    370
    Name: class, dtype: int64

    Division is not exact 1:1, but its good enough
    '''

    # Compile a list of sponsors and words used to express sponsor name
    sponsors_list = df['Sponsor name'].unique()
    for a in range(0, sponsors_list.__len__()):
        sponsors_list[a] = str(sponsors_list[a])
    df.drop(columns=['Sponsor name'], inplace=True)

    df.to_excel(OUT_CLEAN, index=False)
    w = open(OUT_SPONSORS, 'w')
    w.write('\n'.join(sponsors_list))
    w.close()

if __name__ == '__main__':

    print('Running clean_data...')
    main()
