import os
from dotenv import load_dotenv
from pathlib import Path
from youtube_api import YouTubeDataAPI
from youtube_transcript_api import YouTubeTranscriptApi
import youtube_dl
import urllib

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

def print_video_infos(video_response):
    items = video_response.get("items")[0]
    # get the snippet, statistics & content details from the video response
    snippet         = items["snippet"]
    statistics      = items["statistics"]
    content_details = items["contentDetails"]
    # get infos from the snippet
    channel_title = snippet["channelTitle"]
    title         = snippet["title"]
    description   = snippet["description"]
    publish_time  = snippet["publishedAt"]
    # get stats infos
    comment_count = statistics["commentCount"]
    like_count    = statistics["likeCount"]
    dislike_count = statistics["dislikeCount"]
    view_count    = statistics["viewCount"]
    # get duration from content details
    duration = content_details["duration"]
    # duration in the form of something like 'PT5H50M15S'
    # parsing it to be something like '5:50:15'
    parsed_duration = re.search(f"PT(\d+H)?(\d+M)?(\d+S)", duration).groups()
    duration_str = ""
    for d in parsed_duration:
        if d:
            duration_str += f"{d[:-1]}:"
    duration_str = duration_str.strip(":")
    print(f"""\
    Title: {title}
    Description: {description}
    Channel Title: {channel_title}
    Publish time: {publish_time}
    Duration: {duration_str}
    Number of comments: {comment_count}
    Number of likes: {like_count}
    Number of dislikes: {dislike_count}
    Number of views: {view_count}
    """)

def main():
    print("Starting main() function")

    print("GOOGLE API KEY loaded as follows: {}\n".format(API_KEY))

    # get a response via YoutubeTranscriptApi.get_transcript(video id)
    yt = GLOBAL_YT
    chanFiles = os.listdir(VID_REPO)

    for chanFile in chanFiles:
        # get videos from each channel
        vids = getVideoIDs(f"{VID_REPO}/{chanFile}")
        print(f"{chanFile}: {len(vids)}")

        for i in range(1):
            video_id = vids[i]

            print(f"video id: {video_id}")
            captions = YouTubeTranscriptApi.get_transcript(video_id)

            # grab video metadata
            part_opts = ['contentDetails', 'fileDetails', 'statistics', 'snippet']
            params = {
                'key': API_KEY,
                'part': '%2C'.join(part_opts),
                'id': video_id,
            }
            uri = urllib.parse.urlencode(params)
            full_url = YT_VIDEOS_ENDPOINT + uri

            # Get the response


        print("\n")





    # Work with each caption in response


if __name__ == '__init__':
    # load env variables
    dotenv_path = Path(DOTENV_PATH)
    load_dotenv()


if __name__ == '__main__':
    print("Starting py-db-populator.. ")
    main()