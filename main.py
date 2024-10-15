import logging
from typing import Optional, List
import os
from dotenv import load_dotenv
from youtube_service import YouTubeService
from DeepInfraAudioClient import Segment

load_dotenv()

# Replace with your actual DeepInfra and Gemini Pro API keys
deepinfra_api_key = os.environ.get("DEEPINFRA_API_KEY")
gemini_pro_api_key = os.environ.get("GEMINI_PRO_API_KEY")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ['LITELLM_LOG'] = 'INFO'

class VideoInfo:
    def __init__(self, yt_dlp_info: dict, id: str, title: str, description: str, filename: str, ext: str, chapters: List[dict]):
        self.yt_dlp_info: dict = yt_dlp_info
        self.id: str = id
        self.title: str = title
        self.description: str = description
        self.filename: str = filename
        self.ext: str = ext
        self.chapters: List[dict] = chapters
        self.transcription_orig: Optional[str] = None
        self.transcription_translated: Optional[str] = None
        self.transcription_by_segments: Optional[List[Segment]] = None
        self.summary: Optional[str] = None

def main():
    youtube_service = YouTubeService(deepinfra_api_key, gemini_pro_api_key)
    youtube_link = input("Enter the YouTube video link: ")
    language = input("Enter the target language for the summary (e.g., English): ")
    
    try:
        audio_file = youtube_service.download_audio(youtube_link)
        logger.info("Audio downloaded successfully. Starting transcription...")
        
        transcription = youtube_service.transcribe_audio(audio_file)
        
        if transcription:
            logger.info("Transcription complete. Creating VideoInfo...")
            video_info = VideoInfo(
                yt_dlp_info={},  # You may want to populate this with actual yt-dlp info
                id=os.path.splitext(os.path.basename(audio_file))[0],
                title="",  # You may want to get this from yt-dlp info
                description="",  # You may want to get this from yt-dlp info
                filename=audio_file,
                ext=os.path.splitext(audio_file)[1][1:],
                chapters=[]  # You may want to get this from yt-dlp info
            )
            video_info.transcription_orig = transcription
            
            logger.info("Starting summarization...")
            summary = youtube_service.summarize_video(video_info, target_language=language)
            
            if summary:
                logger.info("Summarization complete. Saving to files...")
                youtube_service.save_to_file(transcription, "transcription.txt")
                youtube_service.save_to_file(summary, "summary.md")
                logger.info("Files saved successfully.")
            else:
                logger.info("Summarization failed.")
        else:
            logger.info("Transcription failed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
