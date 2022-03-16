import os
import urllib
import urllib3
import csv
import json

import youtube_dl
from dotenv import load_dotenv
from pathlib import Path
from youtube_api import YouTubeDataAPI
from youtube_transcript_api import YouTubeTranscriptApi

TARGET_CHANNELS = [
    'halfasinteresting'
]

DOTENV_PATH = '.env.local'
API_KEY = os.getenv('GOOGLE_API_KEY')
GLOBAL_YT = YouTubeDataAPI(API_KEY)
VID_REPO = 'vid_repo'

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

def getVideoIDs(filename):
    res = []
    with open(filename, 'r') as f:
        for x in f:
            res.append(x)
    return res

class Entry:
    def __init__(self, video_id, captions):
        self.video_id = video_id
        self.captions = captions

def main():
    print("Starting main() function")

    print("GOOGLE API KEY loaded as follows: {}\n".format(API_KEY))

    # get a response via YoutubeTranscriptApi.get_transcript(video id)
    yt = GLOBAL_YT
    chanFiles = os.listdir(VID_REPO)[1:]

    # To aggregate the entries
    entries = []

    for chanFile in chanFiles:
        # get videos from each channel
        vids = getVideoIDs(f"{VID_REPO}/{chanFile}")
        print(f"{chanFile}: {len(vids)}")

        for i in range(2): #range(len(vids)):
            video_id = vids[i]

            try:
                captions = YouTubeTranscriptApi.get_transcript(video_id)
            except:
                continue

            entries.append({'vid': video_id, 'transcript': captions})
            print(f'{chanFile}: ({i+1}/{len(vids)})`')

            # grab video metadata TODO some other day
            part_opts = ['contentDetails', 'statistics', 'snippet']
            print('%2C'.join(part_opts))
            params = {
                'part': 'snippet',
                'id': video_id,
                'key': API_KEY
            }
            uri = urllib.parse.urlencode(params)
            full_url = YT_VIDEOS_ENDPOINT + uri
            print("full_uri: ", full_url)

            # Get the response from the API
            http = urllib3.PoolManager()
            r = http.request('GET', full_url)

            print(r.data)

    print(f'Channels completed parsing; {len(entries)} videos were parsed')
    # Work with csv now to output
    with open('output/vid_captions.csv', 'w') as f:
        writer = csv.writer(f)

        for entry in entries:
            # encode the captions
            # substep, convert time/caption pairs into a single string
            captions = []
            for caption in entry['transcript']:
               captions.append(json.dumps(caption)) 

            encodedCaptions = f"[{','.join(captions)}]"
            writer.writerow([entry['vid'], encodedCaptions])


if __name__ == '__init__':
    # load env variables
    dotenv_path = Path(DOTENV_PATH)
    load_dotenv()


if __name__ == '__main__':
    print("Starting py-db-populator.. ")
    main()