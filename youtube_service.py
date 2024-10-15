import logging
from typing import Optional, List
import os
import yt_dlp
from litellm import completion, completion_cost
from litellm.types.utils import ModelResponse

from DeepInfraAudioClient import DeepInfraAudioClient, Segment
from SRTFile import SRTFile, TimeCode, SubtitleEntry
from main import VideoInfo

logger = logging.getLogger(__name__)

class YouTubeService:
    def __init__(self, deepinfra_api_key: str, gemini_pro_api_key: str):
        self.deepinfra_api_key = deepinfra_api_key
        self.gemini_pro_api_key = gemini_pro_api_key

    @staticmethod
    def download(youtube_link: str, keep_original: bool = True) -> VideoInfo:
        """
        Downloads the audio from a YouTube video using yt-dlp.

        :param youtube_link: URL of the YouTube video
        :param keep_original: If True, keeps the original audio format if it's WebM
        :return: VideoInfo object containing information about the downloaded video
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
            ext = video.get('ext', 'webm' if keep_original else 'mp3')
            filename = f"{id}.{ext}"

            logger.info(f"Downloaded file: {filename}")
            logger.info(f"Video title: {video.get('title', '')}")

            if os.path.isfile(filename):
                return VideoInfo(
                    yt_dlp_info=video,
                    id=id,
                    title=video.get('title', ''),
                    description=video.get('description', ''),
                    filename=filename,
                    ext=ext,
                    chapters=video.get('chapters', [])
                )
            else:
                raise FileNotFoundError("Audio file not found.")

    def transcribe_audio(self, video_info: VideoInfo) -> VideoInfo:
        """Transcribes the audio file using the DeepInfra API (Whisper large turbo model) via LiteLLM."""
        client = DeepInfraAudioClient(api_key=self.deepinfra_api_key)
        initial_prompt = self.generate_initial_prompt(video_info.filename)
        response = client.transcribe(video_info.filename, initial_prompt=initial_prompt)
        logger.info(f"Segments count: {len(response.segments)}")
        logger.info(f"Transcription cost: ${response.inference_status.cost}, Input tokens: {response.inference_status.tokens_input}, Output tokens: {response.inference_status.tokens_generated}, Runtime: {response.inference_status.runtime_ms} ms")
        video_info.transcription_orig = response.text
        video_info.transcription_by_segments = response.segments
        return video_info

    def generate_initial_prompt(self, audio_file: str) -> str:
        """Generates an initial prompt for the Whisper model using Gemini."""
        system_prompt = """
        You are an AI assistant tasked with generating an initial prompt for a speech recognition model. 
        The prompt should provide context and guidance to improve the accuracy of the transcription. 
        Consider the following aspects:
        1. The likely content based on the file name
        2. Potential technical terms or jargon that might be used
        3. The structure of typical content in this domain (e.g., lectures, interviews, presentations)
        
        Your prompt should be concise but informative, helping the model to better understand the context of the audio it's about to transcribe.
        """

        user_prompt = f"Generate an initial prompt for a speech recognition model that's about to transcribe an audio file named '{audio_file}'."

        try:
            response: ModelResponse = completion(
                model="gemini/gemini-1.5-pro-002",
                api_key=self.gemini_pro_api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating initial prompt: {e}")
            return ""

    def summarize_video(self, video_info: 'VideoInfo', target_language: str) -> Optional[str]:
        """Summarizes the transcription using the Gemini Pro API via LiteLLM."""
        system_prompt = f"""
        You are an AI assistant tasked with summarizing a given text and translating the summary into a specified target language in Markdown format. Follow these steps carefully:
        """

        user_prompt = f"""

        Title:        
        <title>
        {video_info.title}
        </title>

        <description>
        {video_info.description}
        </description>

        1. Read and analyze the following transcription:
        <transcription>
        {video_info.transcriptionOrig}
        </transcription>

        2. Read and analyze the following chapters:
        <chapters>
        {video_info.chapters}
        </chapters>

        3. The target language for translation is:
        <target_language>
        {target_language}
        </target_language>

        4. Create a concise summary of the main points and key ideas from the transcription. Your summary should:
           - Capture the essential information
           - Maintain the core message and important details
           - Exclude minor or irrelevant information

        5. Translate your summary into the target language. Ensure that your translation:
           - Accurately conveys the meaning of the summary
           - Is culturally appropriate and natural-sounding in the target language

        6. Present your translated summary in Markdown format, enclosed within <summary> tags. The entire summary should be in the target language.

        Remember to focus on clarity and conciseness in your summary, and accuracy in your translation. Do not include any text in the original language in your final output.
        """

        summary_response: ModelResponse = completion(
            model="gemini/gemini-1.5-pro-002",
            api_key=self.gemini_pro_api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        tokens_used = summary_response.usage.total_tokens
        prompt_tokens = summary_response.usage.prompt_tokens
        completion_tokens = summary_response.usage.completion_tokens
        cost = completion_cost(completion_response=summary_response)

        logger.debug(
            f"Response cost: ${cost}, Total tokens: {tokens_used}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")

        summary = self.extract_summary(summary_response.choices[0].message.content)

        return summary

    @staticmethod
    def extract_summary(content: str) -> str:
        """Extracts the summary from the completion content."""
        summary_start = content.find("<summary>")
        if summary_start == -1:
            return content
        summary_end = content.find("</summary>")
        return content[summary_start+len("<summary>"):summary_end].strip()

    @staticmethod
    def save_to_file(text: str, filename: str) -> None:
        """Saves the given text to a file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
