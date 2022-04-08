from dotenv import load_dotenv
import json
import os
import pandas as pd
from pathlib import Path
import time
from youtube_api import YouTubeDataAPI
from youtube_transcript_api import YouTubeTranscriptApi

DOTENV_PATH = '.env'
dotenv_path = Path(DOTENV_PATH)
load_dotenv()

API_KEY = os.getenv('YT_API_KEY')
GLOBAL_YT = YouTubeDataAPI(API_KEY)
INPUT_PATH = 'data/clean_data.xlsx'
OUTPUT_PATH = 'data/transcripts.xlsx'
FAIL_VID_PATH = 'data/fail_vid.txt'

YT_VIDEOS_ENDPOINT = 'https://www.googleapis.com/youtube/v3/videos?'

############### START YOUTUBE_DL 

class MyLogger(object):

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):

        print(msg)

def my_hook(d):

    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

YDL_OPTS = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'logger': MyLogger(),
    'progress_hooks': [my_hook],
}

############### END YOUTUBE_DL 

class Entry:

    def __init__(self, video_id, captions):

        self.video_id = video_id
        self.captions = captions

def main():

    print("YT_API_KEY loaded as follows: {}".format(API_KEY))

    df = pd.read_excel(INPUT_PATH, engine='openpyxl')
    vid_ids = list(df['vid_id'])
    classes = list(df['class'])
    fail_vid_ids = []
    fail_count = 0
    check = 0
    total_time = 0

    df = pd.DataFrame()
    df['vid_id'] = ''
    df['class'] = ''
    df['transcript_obj'] = ''

    # Download transcripts and append to df
    print('Downloading transcripts...')
    for a in range(0, len(vid_ids)):
        startTime = time.time()
        try:
            raw_captions = YouTubeTranscriptApi.get_transcript(vid_ids[a])
            captions = []
            for raw_c in raw_captions:
                captions.append(json.dumps(raw_c))
            new_row = pd.DataFrame({
                'vid_id': vid_ids[a],
                'class': classes[a],
                'transcript_obj': '[' + ','.join(captions) + ']'
            })
            df = pd.concat([df, new_row], ignore_index=True)
            # VERY IMPORTANT, otherwise n^2 data will be wasted
            del new_row
        except: # Keep count of errors
            fail_count += 1
            fail_vid_ids.append(vid_ids[a])
        # Print the estimated time of completion (not the most accurate)
        if ((a+1)% 10 == 0):
            total_time += (time.time() - startTime)
            check += 1
            print(f'({a+1}/{len(vid_ids)}) ETC: {((len(vid_ids)-(a+1))*total_time/(check))} s')

    # Print any necessary statistics
    print(f'Failed to fetch: {fail_count}/{len(vid_ids)}')
    print(df['class'].value_counts())

    df.to_excel(OUTPUT_PATH, index=False)
    fail_vid_ids = '\n'.join(fail_vid_ids)
    w = open(FAIL_VID_PATH, 'w')
    w.write(fail_vid_ids)
    w.close()

if __name__ == '__main__':

    print("Starting get_transcripts.. ")
    main()
