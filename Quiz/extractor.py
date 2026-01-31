from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

ytt_api = YouTubeTranscriptApi()

# Get Video ID
def getID(url):
    id = url.split('=')
    return id[-1]

# Get Transcript
def get_transript(video_id):
    try:
        return ' '.join([transcript.text for transcript in ytt_api.fetch(video_id)])
    except TranscriptsDisabled:
        return "This video has no transcripts enabled."