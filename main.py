import logging
from dataclasses import dataclass
from typing import Optional

import litellm
import yt_dlp
import os
from litellm import completion, transcription
from litellm.types.utils import ModelResponse, TranscriptionResponse
from dotenv import load_dotenv

from DeepInfraAudioClient import DeepInfraAudioClient, Segment

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

# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
litellm.register_model({
    "deepinfra/openai/whisper-large-v3-turbo": {
        "mode": "audio_transcription",
        "input_cost_per_second": 0.00000333333,
        "output_cost_per_second": 0.00000333333,
        "litellm_provider": "openai"
    },
})

@dataclass
class VideoInfo:
    ytDlpInfo: dict
    id: str
    title: str
    description: str
    filename: str
    ext: str
    chapters: list
    transcriptionOrig: Optional[str]
    transcriptionTranslated: Optional[str]
    transcriptionBySegments: Optional[list[Segment]]
    summary: Optional[str]

def download_audio(youtube_link, keep_original=True):
    """
    Downloads the audio from a YouTube video using yt-dlp.

    :param youtube_link: URL of the YouTube video
    :param keep_original: If True, keeps the original audio format if it's WebM
    :return: Path to the downloaded file
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'noplaylist': True
    }

    if not keep_original:
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
        ydl_opts['extract_audio'] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_link, download=True)
        if 'entries' in info:
            video = info['entries'][0]
        else:
            video = info

        id = video.get('id', '')
        title = video.get('title', '')
        description = video.get('description', '')
        ext = video.get('ext', 'webm' if keep_original else 'mp3')
        filename = f"{id}.{ext}"

        chapters = video.get('chapters', None)
        print(f"Chapters: {chapters}")
        print(f"Downloaded file: {filename}")
        print(f"Video title: {title}")

        if os.path.isfile(filename):
            return filename
        else:
            raise FileNotFoundError("Audio file not found.")


def transcribe_audio(audio_file):
    """Transcribes the audio file using the DeepInfra API (Whisper large turbo model) via LiteLLM."""
    client = DeepInfraAudioClient(api_key=deepinfra_api_key)
    response = client.transcribe(audio_file)
    logger.info(f"Segments count: {len(response.segments)}")
    logger.info(f"Transcription cost: ${response.inference_status.cost}, Input tokens: {response.inference_status.tokens_input}, Output tokens: {response.inference_status.tokens_generated}, Runtime: {response.inference_status.runtime_ms} ms")
    return response.text

def extract_summary(content):
    """Extracts the summary from the completion content."""
    summary_start = content.find("<summary>")
    if summary_start == -1:
        return content
    summary_end = content.find("</summary>")
    return content[summary_start+len("<summary>"):summary_end].strip()


def summarize_video(videoinfo, target_language):
    """Summarizes the transcription using the Gemini Pro API via LiteLLM."""
    try:
        system_prompt = f"""
You are an AI assistant tasked with summarizing a given text and translating the summary into a specified target language in Markdown format. Follow these steps carefully:
"""

        user_prompt = f"""

Title:        
<title>
{title}
</title>

<description>
{description}
</description>

1. Read and analyze the following transcription:
<transcription>
{transcription}
</transcription>

2. Read and analyze the following chapters:
<chapters>
{chapters}
</chapters>

3. The target language for translation is:
<target_language>
{target_language}
</target_language>

3. Create a concise summary of the main points and key ideas from the transcription. Your summary should:
   - Capture the essential information
   - Maintain the core message and important details
   - Exclude minor or irrelevant information

4. Translate your summary into the target language. Ensure that your translation:
   - Accurately conveys the meaning of the summary
   - Is culturally appropriate and natural-sounding in the target language

5. Present your translated summary in Markdown format, enclosed within <summary> tags. The entire summary should be in the target language.

Remember to focus on clarity and conciseness in your summary, and accuracy in your translation. Do not include any text in the original language in your final output.
"""

        summary_response: ModelResponse = litellm.completion(
            model="gemini/gemini-1.5-pro-002",
            api_key=gemini_pro_api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        tokens_used = summary_response.usage.total_tokens
        prompt_tokens = summary_response.usage.prompt_tokens
        completion_tokens = summary_response.usage.completion_tokens
        cost = litellm.completion_cost(completion_response=summary_response)

        logger.debug(
            f"Response cost: ${cost}, Total tokens: {tokens_used}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")

        summary = extract_summary(summary_response.choices[0].message.content)

        return summary
    except Exception as e:
        logger.info(f"Error during summarization: {e}")
        return None


def save_to_file(text, filename):
    """Saves the given text to a file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.info(f"Error saving to file: {e}")

def main():
    youtube_link = input("Enter the YouTube video link: ")
    language = input("Enter the target language for the summary (e.g., English): ")
    audio_file = download_audio(youtube_link)

    if audio_file:
        logger.info("Audio downloaded successfully. Starting transcription...")
        transcription = transcribe_audio(audio_file)

        if transcription:
            logger.info("Transcription complete. Starting summarization...")
            summary = summarize_video(transcription, target_language=language)

            if summary:
                logger.info("Summarization complete. Saving to files...")
                save_to_file(transcription, "transcription.txt")
                save_to_file(summary, "summary.md")
                logger.info("Files saved successfully.")
            else:
                logger.info("Summarization failed.")
        else:
            logger.info("Transcription failed.")
    else:
        logger.info("Audio download failed.")


if __name__ == "__main__":
    main()