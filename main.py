import logging
import os

from dotenv import load_dotenv

from youtube_service import YouTubeService

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



def main():
    youtube_service = YouTubeService(deepinfra_api_key, gemini_pro_api_key)
    youtube_link = input("Enter the YouTube video link: ")
    language = input("Enter the target language for the summary (e.g., English): ")

    try:
        video_info = youtube_service.download(youtube_link)
        logger.info("Audio downloaded successfully")

        video_info = youtube_service.transcribe_audio(video_info)

        if video_info.transcription_orig:
            logger.info("Transcription complete. Starting summarization...")
            summary = youtube_service.summarize_video(video_info, target_language=language)

            if summary:
                logger.info("Summarization complete. Saving to files...")
                youtube_service.save_transcription_and_summary(video_info, summary)
                logger.info("Files saved successfully.")
            else:
                logger.info("Summarization failed.")
        else:
            logger.info("Transcription failed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
