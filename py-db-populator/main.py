import os
from dotenv import load_dotenv
from pathlib import Path
from youtube_api import YoutubeDataAPI
import youtube_dl

TARGET_CHANNELS = [
    'halfasinteresting'
]

DOTENV_PATH = '.env.local'
API_KEY = os.getenv('GOOGLE_API_KEY')
GLOBAL_YT = YoutubeDataAPI(API_KEY)

#channame string
def findAllVideosForChan(channame):


def main():
    print("Starting main() function")

    print("GOOGLE API KEY loaded as follows: {}\n".format(API_KEY))

    # Find the video id of the videos on a certain channel
    for chan in TARGET_CHANNELS:
        print('Finding video id\'s for {}'.format(chan))


    # get a response via YoutubeTranscriptApi.get_transcript(video id)


    # Work with each caption in response


if __name__ == '__init__':
    # load env variables
    dotenv_path = Path(DOTENV_PATH)
    load_dotenv()


if __name__ == '__main__':
    print("Starting py-db-populator.. ")
    main()