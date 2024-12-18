# main.py

import argparse
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union, Dict

import backoff
import litellm
import requests
import webvtt
import yt_dlp
from dotenv import load_dotenv
from litellm import completion, completion_cost
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, HttpUrl
from weasyprint import HTML

from audio_splitter import AudioChunksSplitter  # Ensure this module exists

# Load environment variables
load_dotenv()

# Replace with your actual DeepInfra and Gemini Pro API keys
deepinfra_api_key = os.environ.get("DEEPINFRA_API_KEY")
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
INITIAL_PROMPT_MODEL_ID = "openrouter/openai/gpt-4o"
SUMMARY_MODEL_ID = "openrouter/openai/o1-mini" # or "o1" for huge files/videos

# gemini_pro_api_key = os.environ.get("GEMINI_PRO_API_KEY")

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ['LITELLM_LOG'] = 'INFO'

litellm.register_model({
    "openai/o1": {
        "max_tokens": 200000,
        "max_input_tokens": 200000,
        "max_output_tokens": 100000,
        "input_cost_per_token": 1.5e-5,
        "output_cost_per_token": 6e-5,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_system_messages": True,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_response_schema": True,
        "source": "https://openrouter.ai/openai/o1"
    },
})

# Enums and Models
class Task(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class ChunkLevel(str, Enum):
    SEGMENT = "segment"
    WORD = "word"


class InferenceStatus(str, Enum):
    UNKNOWN = "unknown"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    confidence: Optional[float] = None
    words: Optional[List[Word]] = None


class InferenceReplyStatus(BaseModel):
    status: InferenceStatus = InferenceStatus.SUCCEEDED
    runtime_ms: int = 0
    cost: float
    tokens_generated: Optional[int] = None
    tokens_input: Optional[int] = None


class AutomaticSpeechRecognitionOut(BaseModel):
    text: str
    segments: Optional[List[Segment]] = None
    language: Optional[str] = None
    input_length_ms: int = 0
    request_id: Optional[str] = None
    inference_status: Optional[InferenceReplyStatus] = None


# Custom Exception
class DeepInfraAudioClientError(Exception):
    """Custom exception for DeepInfraAudioClient errors"""


# DeepInfra Audio Client
class DeepInfraAudioClient:
    BASE_URL: str = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3-turbo"

    def __init__(self, api_key: str):
        self.api_key: str = api_key
        self.session: requests.Session = requests.Session()
        self.session.headers.update({"Authorization": f"bearer {api_key}"})

    def transcribe(
            self,
            audio_file: Union[str, bytes],
            task: Task = Task.TRANSCRIBE,
            initial_prompt: Optional[str] = None,
            temperature: float = 0,
            language: Optional[str] = None,
            chunk_level: ChunkLevel = ChunkLevel.SEGMENT,
            chunk_length_s: int = 30,
            webhook: Optional[HttpUrl] = None
    ) -> AutomaticSpeechRecognitionOut:
        """
        Transcribe an audio file using the DeepInfra API.

        Args:
            audio_file (Union[str, bytes]): Path to the audio file or bytes content of the audio file.
            task (Task, optional): Task to perform. Defaults to Task.TRANSCRIBE.
            initial_prompt (str, optional): Optional text to provide as a prompt for the first window.
            temperature (float, optional): Temperature to use for sampling. Defaults to 0.
            language (str, optional): Language that the audio is in (ISO 639-1 code).
            chunk_level (ChunkLevel, optional): Chunk level. Defaults to ChunkLevel.SEGMENT.
            chunk_length_s (int, optional): Chunk length in seconds to split audio. Defaults to 30.
            webhook (HttpUrl, optional): The webhook to call when inference is done.

        Returns:
            AutomaticSpeechRecognitionOut: The transcription result.

        Raises:
            DeepInfraAudioClientError: If there's an error during the API request or processing.
        """
        try:
            files: Dict[str, Union[str, bytes]] = self._prepare_audio_file(audio_file)
            data: Dict[str, Union[str, float, int]] = self._prepare_request_data(
                task, initial_prompt, temperature, language, chunk_level, chunk_length_s, webhook
            )

            response: requests.Response = self.session.post(self.BASE_URL, files=files, data=data)
            response.raise_for_status()

            return AutomaticSpeechRecognitionOut(**response.json())
        except requests.RequestException as e:
            response_text = e.response.text if e.response else "No response text"
            raise DeepInfraAudioClientError(f"Error during API request: {str(e)}, response: {response_text}") from e
        except Exception as e:
            raise DeepInfraAudioClientError(f"Unexpected error: {str(e)}") from e

    def _prepare_audio_file(self, audio_file: Union[str, bytes]) -> Dict[str, Union[str, bytes]]:
        if isinstance(audio_file, str):
            if not os.path.isfile(audio_file):
                raise DeepInfraAudioClientError(f"Audio file not found: {audio_file}")
            with open(audio_file, "rb") as file:
                return {"audio": file.read()}
        elif isinstance(audio_file, bytes):
            return {"audio": audio_file}
        else:
            raise DeepInfraAudioClientError("Invalid audio_file type. Expected str or bytes.")

    def _prepare_request_data(
            self,
            task: Task,
            initial_prompt: Optional[str],
            temperature: float,
            language: Optional[str],
            chunk_level: ChunkLevel,
            chunk_length_s: int,
            webhook: Optional[HttpUrl]
    ) -> Dict[str, Union[str, float, int]]:
        data: Dict[str, Union[str, float, int]] = {
            "task": task.value,
            "temperature": temperature,
            "chunk_level": chunk_level.value,
            "chunk_length_s": chunk_length_s,
        }

        if initial_prompt:
            data["initial_prompt"] = initial_prompt
        if language:
            data["language"] = language
        if webhook:
            data["webhook"] = str(webhook)

        return data


# Video Information Class
class VideoInfo:
    def __init__(self, yt_dlp_info: Optional[dict], id: str, title: str, description: str, filename: str, ext: str,
                 language: Optional[str],
                 chapters: List[dict]):
        self.yt_dlp_info: Optional[dict] = yt_dlp_info
        self.id: str = id
        self.title: str = title
        self.description: str = description
        self.filename: str = filename
        self.ext: str = ext
        self.language: Optional[str] = language
        self.chapters: List[dict] = chapters
        self.transcription_orig: Optional[str] = None
        self.transcription_translated: Optional[str] = None
        self.transcription_by_segments: Optional[List[Segment]] = None
        self.summary: Optional[str] = None


# Dataclasses for TimeCode and Subtitles
@dataclass
class TimeCode:
    hours: int
    minutes: int
    seconds: int
    milliseconds: int

    def __str__(self) -> str:
        return f"{self.hours:02}:{self.minutes:02}:{self.seconds:02},{self.milliseconds:03}"

    @classmethod
    def from_string(cls, time_string: str) -> 'TimeCode':
        match = re.match(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})", time_string)
        if not match:
            raise ValueError(f"Invalid time format: {time_string}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))

    @classmethod
    def from_seconds(cls, seconds: float) -> 'TimeCode':
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        whole_seconds = int(seconds)
        milliseconds = int(round((seconds - whole_seconds) * 1000))
        return cls(int(hours), int(minutes), whole_seconds, milliseconds)


@dataclass
class SubtitleEntry:
    """
    Represents a single subtitle entry in an SRT file.
    """
    index: int  # From 1
    start_time: TimeCode
    end_time: TimeCode
    text: List[str]

    def __str__(self) -> str:
        subtitle_text = "\n".join(self.text)
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{subtitle_text}\n"

    @classmethod
    def from_string(cls, entry_string: str) -> 'SubtitleEntry':
        parts = entry_string.strip().split("\n")
        if len(parts) < 3:
            raise ValueError("Invalid subtitle entry")

        index = int(parts[0])
        time_range = parts[1]
        start_time_str, end_time_str = time_range.split(" --> ")
        start_time = TimeCode.from_string(start_time_str)
        end_time = TimeCode.from_string(end_time_str)
        text = parts[2:]

        return cls(index, start_time, end_time, text)

@dataclass
class SRTFile:
    subtitles: List[SubtitleEntry] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(str(subtitle) for subtitle in self.subtitles)

    def add_subtitle(self, index: int, start_time: TimeCode, end_time: TimeCode, text: List[str]) -> None:
        subtitle = SubtitleEntry(index=index, start_time=start_time, end_time=end_time, text=text)
        self.subtitles.append(subtitle)

    @classmethod
    def from_string(cls, srt_content: str) -> 'SRTFile':
        entries = srt_content.strip().split("\n\n")
        subtitles = [SubtitleEntry.from_string(entry) for entry in entries]
        return cls(subtitles)

    def to_file(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(self))

    @classmethod
    def from_file(cls, filename: str) -> 'SRTFile':
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return cls.from_string(content)


# Transcription Service
class TranscriptionService:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.client = DeepInfraAudioClient(api_key=api_key)
        self.max_retries = max_retries

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        giveup=lambda e: isinstance(e, ValueError)
    )
    def transcribe_chunk(self, audio_chunk, language: Optional[str] = None,
                         initial_prompt: Optional[str] = None) -> AutomaticSpeechRecognitionOut:
        """Transcribe a single audio chunk with retries"""
        return self.client.transcribe(audio_chunk, language=language, initial_prompt=initial_prompt)

    def transcribe_audio_file_splitted(self, audio_file: str, language: Optional[str] = None,
                                       initial_prompt: Optional[str] = None) -> AutomaticSpeechRecognitionOut:
        """Split audio file into chunks and transcribe each chunk"""
        # Split audio into chunks
        chunks = AudioChunksSplitter.load_and_chunk_audio(audio_file)
        logger.info(f"Split audio into {len(chunks)} chunks")

        # Transcribe each chunk
        results: List[AutomaticSpeechRecognitionOut] = []
        total_cost = 0
        total_runtime = 0

        for i, chunk in enumerate(chunks):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunks)}")
            try:
                chunk_result = self.transcribe_chunk(chunk, language, initial_prompt)
                results.append(chunk_result)

                # Accumulate metrics
                if chunk_result.inference_status:
                    total_cost += chunk_result.inference_status.cost
                    total_runtime += chunk_result.inference_status.runtime_ms

                logger.info(f"Chunk {i + 1} transcribed successfully")
                logger.debug(f"Chunk {i + 1} cost: ${chunk_result.inference_status.cost:.4f}")

            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i + 1}: {str(e)}")
                raise

        # Combine results
        combined_result = self._combine_results(results)
        if combined_result.inference_status:
            combined_result.inference_status.cost = total_cost
            combined_result.inference_status.runtime_ms = total_runtime

        return combined_result

    def _combine_results(self, results: List[AutomaticSpeechRecognitionOut]) -> AutomaticSpeechRecognitionOut:
        """Combine multiple transcription results into one"""
        if not results:
            raise ValueError("No results to combine")

        combined = results[0]

        if combined.segments is None:
            combined.segments = []

        if combined.inference_status is None:
            combined.inference_status = InferenceReplyStatus(cost=0, runtime_ms=0)

        for result in results[1:]:
            combined.text += " " + result.text
            if result.segments:
                last_end = combined.segments[-1].end if combined.segments else 0
                for segment in result.segments:
                    segment.start += last_end
                    segment.end += last_end
                    combined.segments.append(segment)

        return combined

    def transcribe_audio(self, audio_file: str, language: Optional[str] = None,
                         initial_prompt: Optional[str] = None) -> AutomaticSpeechRecognitionOut:
        """Transcribes the audio file using the DeepInfra API (Whisper large turbo model) via LiteLLM."""
        response = self.client.transcribe(audio_file, language=language, initial_prompt=initial_prompt)
        return response


# Transcription Database
class TranscriptionDatabase:
    def __init__(self, db_name: str = 'transcriptions.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                video_id TEXT PRIMARY KEY,
                transcription TEXT,
                language TEXT,
                segments TEXT
            )
        ''')
        self.conn.commit()

    def save_transcription(self, video_id: str, transcription: str, language: str, segments: str):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO transcriptions (video_id, transcription, language, segments)
            VALUES (?, ?, ?, ?)
        ''', (video_id, transcription, language, segments))
        self.conn.commit()

    def get_transcription(self, video_id: str) -> Optional[tuple[str, str, str]]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT transcription, language, segments FROM transcriptions WHERE video_id = ?', (video_id,))
        result = cursor.fetchone()
        return result if result else None

    def close(self):
        self.conn.close()


# Utility Functions
def extract_prompt(content: str) -> str:
    """
        Extract the prompt from the content that starts with:
        ```xml
        <prompt>
    """
    prompt_start = content.find("<prompt>")
    if prompt_start == -1:
        return content
    prompt_end = content.find("</prompt>")
    return content[prompt_start + len("<prompt>"):prompt_end].strip()


def _segments_to_srt(segments: List[Segment]) -> str:
    """Converts transcription segments to SRT format."""
    srt_file = SRTFile()

    for idx, segment in enumerate(segments, 1):
        start_time = TimeCode.from_seconds(segment.start)
        end_time = TimeCode.from_seconds(segment.end)

        srt_file.add_subtitle(idx, start_time, end_time, [segment.text])

    return str(srt_file)


# YouTube Service
class YouTubeService:
    def __init__(self, use_subtitles_from_youtube: bool = False):
        self.USE_SUBTITLES_FROM_YOUTUBE = use_subtitles_from_youtube
        self.db = TranscriptionDatabase()

    @staticmethod
    def download(youtube_link: str, keep_original: bool = True, download_subtitles: bool = False) -> VideoInfo:
        """
        Downloads the audio from a YouTube video using yt-dlp.

        :param youtube_link: URL of the YouTube video
        :param keep_original: If True, keeps the original audio format if it's WebM
        :param download_subtitles: If True, attempts to download subtitles
        :return: VideoInfo object containing information about the downloaded video
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            'noplaylist': True,
            'writesubtitles': download_subtitles,
            'subtitleslangs': ['en'],  # Modify as needed or make dynamic
            'skip_download': not download_subtitles  # If downloading subtitles only
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
                    title=video.get('fulltitle', video.get('title', '')),
                    description=video.get('description', ''),
                    filename=filename,
                    ext=ext,
                    language=video.get('language', ''),
                    chapters=video.get('chapters', [])
                )
            else:
                raise FileNotFoundError("Audio file not found.")

    def download_subtitles(self, youtube_link: str) -> Optional[List[SubtitleEntry]]:
        """
        Downloads subtitles from YouTube if available.

        :param youtube_link: URL of the YouTube video
        :return: List of SubtitleEntry objects or None if not available
        """
        ydl_opts = {
            'writesubtitles': True,
            'subtitleslangs': ['en'],  # Modify as needed or make dynamic
            'skip_download': True,
            'outtmpl': '%(id)s.%(ext)s',
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(youtube_link, download=False)
                subtitles = info.get('subtitles', {})
                if 'en' in subtitles:
                    ydl.download([youtube_link])
                    subtitle_filename = f"{info['id']}.en.vtt"
                    if os.path.isfile(subtitle_filename):
                        logger.info(f"Downloaded subtitles: {subtitle_filename}")
                        return self.vtt_to_srt(subtitle_filename)
                else:
                    logger.info("No English subtitles available.")
                    return None
            except Exception as e:
                logger.error(f"Error downloading subtitles: {e}")
                return None

    @staticmethod
    def vtt_to_srt(vtt_filename: str) -> List[SubtitleEntry]:
        """
        Converts a VTT subtitle file to a list of SRT SubtitleEntry objects.

        :param vtt_filename: Path to the VTT file
        :return: List of SubtitleEntry objects
        """
        srt_entries = []
        try:
            index = 1
            for caption in webvtt.read(vtt_filename):
                start = YouTubeService.vtt_timestamp_to_seconds(caption.start)
                end = YouTubeService.vtt_timestamp_to_seconds(caption.end)
                text = caption.text.replace('\n', ' ').strip()
                srt_entries.append(
                    SubtitleEntry(
                        index=index,
                        start_time=TimeCode.from_seconds(start),
                        end_time=TimeCode.from_seconds(end),
                        text=[text]
                    )
                )
                index += 1
            return srt_entries
        except Exception as e:
            logger.error(f"Error converting VTT to SRT: {e}")
            return []

    @staticmethod
    def vtt_timestamp_to_seconds(timestamp: str) -> float:
        """
        Converts VTT timestamp to seconds.

        :param timestamp: VTT timestamp string
        :return: Time in seconds
        """
        try:
            hours, minutes, seconds = timestamp.split(':')
            seconds, milliseconds = seconds.split('.')
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
            return total_seconds
        except Exception as e:
            logger.error(f"Error parsing timestamp {timestamp}: {e}")
            return 0.0

    def transcribe_audio(self, video_info: VideoInfo) -> VideoInfo:
        """Transcribes the audio file using the DeepInfra API (Whisper large turbo model) via LiteLLM."""
        cached_result = self.db.get_transcription(video_info.id)
        if cached_result:
            logger.info("Using cached transcription")
            transcription, language, segments_json = cached_result
            video_info.transcription_orig = transcription
            video_info.language = language
            try:
                segments_data = json.loads(segments_json)
                video_info.transcription_by_segments = [Segment(**segment) for segment in segments_data]
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding segments JSON: {e}")
                video_info.transcription_by_segments = None
            return video_info

        transcription_service = TranscriptionService(api_key=deepinfra_api_key)
        # initial_prompt = self.generate_initial_prompt(video_info)
        initial_prompt = "" # DeepInfra doesn't support initial promot
        # Gemini Pro costs $0.001875 per second
        logger.debug(f"Transcribing audio file: {video_info.filename}")
        SPLIT_AUDIO = False
        if SPLIT_AUDIO:
            response = transcription_service.transcribe_audio_file_splitted(
                video_info.filename,
                language=video_info.language,
                initial_prompt=initial_prompt
            )
        else:
            response = transcription_service.transcribe_audio(
                video_info.filename,
                language=video_info.language,
                initial_prompt=initial_prompt
            )
        logger.info(f"Segments count: {len(response.segments) if response.segments else 0}")
        logger.info(
            f"Transcription cost: ${response.inference_status.cost if response.inference_status else 0}, "
            f"Input tokens: {response.inference_status.tokens_input if response.inference_status else 0}, "
            f"Output tokens: {response.inference_status.tokens_generated if response.inference_status else 0}, "
            f"Runtime: {response.inference_status.runtime_ms if response.inference_status else 0} ms")

        video_info.transcription_orig = response.text
        video_info.language = response.language  # Ensure language is updated
        video_info.transcription_by_segments = response.segments

        # Cache the transcription and segments
        if response.segments is not None:
            segments_json = json.dumps([segment.model_dump() for segment in response.segments])
        else:
            segments_json = json.dumps([])

        self.db.save_transcription(video_info.id, video_info.transcription_orig, video_info.language, segments_json)

        return video_info

    def generate_initial_prompt(self, video_info: VideoInfo) -> str:
        """Generates an initial prompt for the Whisper model using Gemini."""
        # https://cookbook.openai.com/examples/whisper_prompting_guide

        language = video_info.language
        if language is None:
            language = "English"

        system_prompt = f"""
You are an AI assistant tasked with generating an initial prompt for a speech recognition model. Your goal is to create a prompt that will provide context and guidance to improve the accuracy of the transcription for a given video. Follow these instructions carefully:

1. You will be provided with the following information:
   <video_name>{video_info.title}</video_name>
   <video_description>{video_info.description}</video_description>
   <language>{language}</language>

2. Based on the video name and description, generate a prompt that will help improve the accuracy of the transcription. The prompt should be no longer than 150 tokens (around 100 words).

3. Ensure that the prompt is written entirely in the language specified in the <language> variable.

4. When creating the prompt, follow these best practices:
   a. Provide contextual information about the video, such as the topic, speakers, or type of content.
   b. Include relevant terminology, proper nouns, or technical terms that may appear in the video.
   c. Use proper capitalization, punctuation, and formatting to guide the model's output style.
   d. Keep the prompt concise and focused on providing stylistic guidance and context.
   e. If applicable, include common filler words or speech patterns that may be present in the video.

5. Do not include instructions or commands in the prompt, as the speech recognition model will not execute them.

6. Avoid including specific timestamps or references to video duration in the prompt.

7. Write your generated prompt inside <prompt> tags.

Remember, the goal is to create a prompt that will help the speech recognition model produce a more accurate transcription by providing context and stylistic guidance relevant to the video content.
"""

        user_prompt = f"Generate an initial prompt."

        try:
            response: ModelResponse = completion(
                model=INITIAL_PROMPT_MODEL_ID,
                api_key=openrouter_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=150
            )
            response_prompt = extract_prompt(response.choices[0].message.content)

            words = response_prompt.split()
            if len(words) > 100:
                logger.warning(f"Initial prompt is longer than 100 words. Truncating to 100 words.")
                response_prompt = " ".join(words[:100])

            return response_prompt
        except Exception as e:
            logger.error(f"Error generating initial prompt: {e}")
            return ""

    def summarize_video(self, video_info: 'VideoInfo', target_language: str) -> Optional[str]:
        """Summarizes the transcription using the Gemini Pro API via LiteLLM, returns HTML."""
        system_prompt = f"""
You are an AI assistant specialized in summarizing and translating video content into HTML format.
"""

        user_prompt = f"""
You are an AI assistant specialized in summarizing and translating video content. Your task is to create a detailed, insightful summary of a video and translate it into a specified target language in HTML format.

First, let's examine the video information:

<video_transcription>
{video_info.transcription_orig}
</video_transcription>

<video_title>
{video_info.title}
</video_title>

<video_description>
{video_info.description}
</video_description>

<video_chapters>
{json.dumps(video_info.chapters, indent=2)}
</video_chapters>

The target language for translation is:
<target_language>
{target_language}
</target_language>

Now, please follow these steps carefully:

1. Analyze the video information provided above.

2. Create a detailed summary of the video content. Wrap your thought process in <analysis> tags:
   <analysis>
   - Identify the main topics discussed in the video
   - List key insights and non-obvious ideas presented
   - Note any important facts, figures, or examples
   - Consider the overall structure and flow of the content
   - Determine which elements are essential and which can be omitted
   - Identify the target audience of the video
   - Note any cultural references or context that may need adaptation for the target language
   - List any technical terms or jargon that may require special attention in translation
   - Consider how the video's tone and style should be reflected in the summary
   </analysis>

3. Based on your analysis, write a comprehensive summary that:
   - Captures the essential information and main points
   - Highlights the topics discussed, key insights, and non-obvious ideas
   - Maintains the core message and important details
   - Excludes minor or irrelevant information
   - Is structured in a way that a knowledgeable person would summarize the content

4. Translate your summary into the target language. Ensure that your translation:
   - Accurately conveys the meaning of the summary
   - Is culturally appropriate and natural-sounding in the target language
   - Maintains the structure and emphasis of the original summary
   - Appropriately adapts any cultural references or context
   - Accurately translates technical terms and jargon, providing explanations if necessary

5. Present your translated summary in HTML format, enclosed within <summary> tags. The entire summary should be in the target language.

Here's an example of how your output should be structured (using English as a placeholder):

<summary>
<h1>Video Summary: [Translated Title]</h1>

<h2>Main Topics</h2>
<ul>
  <li>Topic 1</li>
  <li>Topic 2</li>
  <li>Topic 3</li>
</ul>

<h2>Key Insights</h2>
<ul>
  <li>Insight 1</li>
  <li>Insight 2</li>
  <li>Insight 3</li>
</ul>

<h2>Non-obvious Ideas</h2>
<ul>
  <li>Idea 1</li>
  <li>Idea 2</li>
</ul>

<h2>Detailed Summary</h2>
<p>[Your translated, detailed summary goes here, using appropriate HTML formatting for headings, lists, emphasis, etc.]</p>
</summary>

Remember to focus on clarity, conciseness, and accuracy in your summary and translation. Do not include any text in the original language in your final output.
"""

        try:
            response: ModelResponse = completion(
                model=SUMMARY_MODEL_ID,
                api_key=openrouter_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1000  # Adjust as needed
            )
            tokens_used = response.usage.total_tokens
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = completion_cost(completion_response=response)

            logger.debug(
                f"Response cost: ${cost}, Total tokens: {tokens_used}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")

            summary = self.extract_summary(response.choices[0].message.content)

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None

    @staticmethod
    def extract_summary(content: str) -> str:
        """Extracts the summary from the completion content."""
        summary_start = content.find("<summary>")
        if summary_start == -1:
            return content
        summary_end = content.find("</summary>")
        if summary_end == -1:
            return content[summary_start + len("<summary>"):].strip()
        return content[summary_start + len("<summary>"):summary_end].strip()

    @staticmethod
    def save_to_file(text: str, filename: str) -> None:
        """Saves the given text to a file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Error saving to file: {e}")

    def save_transcription_and_summary(self, video_info: VideoInfo, summary_html: str, output_dir: str) -> None:
        """Saves the transcription and summary to files with the video ID in the filename."""
        transcription_filename = os.path.join(output_dir, f"{video_info.id}_transcription.txt")
        summary_html_filename = os.path.join(output_dir, f"{video_info.id}_summary.html")
        summary_pdf_filename = os.path.join(output_dir, f"{video_info.id}_summary.pdf")
        srt_filename = os.path.join(output_dir, f"{video_info.id}_transcription.srt")

        # Save transcription as TXT
        if video_info.transcription_orig:
            self.save_to_file(video_info.transcription_orig, transcription_filename)
            logger.info(f"Transcription saved to {transcription_filename}")

        # Save summary as HTML and PDF
        if summary_html:
            self.save_to_file(summary_html, summary_html_filename)
            logger.info(f"Summary (HTML) saved to {summary_html_filename}")

            try:
                # Convert HTML to PDF using WeasyPrint
                HTML(string=summary_html).write_pdf(summary_pdf_filename)
                logger.info(f"Summary PDF saved to {summary_pdf_filename}")
            except Exception as e:
                logger.error(f"Error converting HTML to PDF: {e}")

        # Save SRT transcription
        if video_info.transcription_by_segments:
            srt_content = _segments_to_srt(video_info.transcription_by_segments)
            self.save_to_file(srt_content, srt_filename)
            logger.info(f"SRT transcription saved to {srt_filename}")



# Main Function
def main():
    parser = argparse.ArgumentParser(description="YouTube Transcription and Summarization Service")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--youtube', type=str, help="YouTube video URL to process")
    group.add_argument('--srt', type=str, help="Path to an existing SRT file to process")

    parser.add_argument('--use-subtitles', action='store_true', help="Use subtitles from YouTube if available", default=True)
    parser.add_argument('--target-language', type=str, required=True,
                        help="Target language for the summary (e.g., English)")
    parser.add_argument('--output-dir', type=str, default='output', help="Directory to save output files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    youtube_service = YouTubeService(
        use_subtitles_from_youtube=args.use_subtitles
    )

    try:
        if args.youtube:
            youtube_link = args.youtube
            language = args.target_language

            if youtube_service.USE_SUBTITLES_FROM_YOUTUBE:
                logger.info("Attempting to download subtitles from YouTube...")
                subtitles = youtube_service.download_subtitles(youtube_link)
                if subtitles:
                    logger.info("Subtitles downloaded successfully. Skipping transcription.")
                    # Create VideoInfo with subtitles
                    video_info = YouTubeService.download(youtube_link, keep_original=True, download_subtitles=False)
                    texts: List[str] = [subtitle.text[0] for subtitle in subtitles]  # Extract text from list
                    video_info.transcription_orig = " ".join(texts)
                    video_info.transcription_by_segments = None  # TODO: Implement if needed
                else:
                    logger.info("No subtitles available. Proceeding with audio transcription.")
                    video_info = youtube_service.download(youtube_link)
                    video_info = youtube_service.transcribe_audio(video_info)
            else:
                # Download audio and transcribe
                video_info = youtube_service.download(youtube_link)
                video_info = youtube_service.transcribe_audio(video_info)

        elif args.srt:
            srt_file_path = args.srt
            language = args.target_language

            if not os.path.isfile(srt_file_path):
                logger.error(f"SRT file not found: {srt_file_path}")
                return

            logger.info(f"Loading SRT file: {srt_file_path}")

            # Parse SRT content into SubtitleEntry objects
            try:
                srt_entries = []
                with open(srt_file_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()

                # Use webvtt to read SRT/VTT files
                if srt_file_path.endswith('.vtt'):
                    srt_entries = youtube_service.vtt_to_srt(srt_file_path)
                else:
                    # Parse standard SRT
                    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2}[.,]\d{3}) --> (\d{2}:\d{2}:\d{2}[.,]\d{3})\n((?:.+\n?)*)',
                                         re.MULTILINE)
                    matches = pattern.findall(srt_content)
                    for match in matches:
                        index_str, start_str, end_str, text = match
                        start_seconds = YouTubeService.vtt_timestamp_to_seconds(start_str.replace(',', '.'))
                        end_seconds = YouTubeService.vtt_timestamp_to_seconds(end_str.replace(',', '.'))
                        text_lines = text.strip().split('\n')
                        srt_entries.append(
                            SubtitleEntry(
                                index=int(index_str),
                                start_time=TimeCode.from_seconds(start_seconds),
                                end_time=TimeCode.from_seconds(end_seconds),
                                text=text_lines
                            )
                        )
            except Exception as e:
                logger.error(f"Error parsing SRT file: {e}")
                return

            # Create a VideoInfo object with the SRT data
            video_info = VideoInfo(
                yt_dlp_info=None,
                id="srt_input",
                title="SRT Input",
                description="Provided via SRT file",
                filename=srt_file_path,
                ext=os.path.splitext(srt_file_path)[1][1:],
                language=language,
                chapters=[]
            )
            texts: List[str] = [subtitle.text[0] for subtitle in srt_entries]  # Extract text from list
            video_info.transcription_orig = " ".join(texts)
            video_info.transcription_by_segments = None  # TODO: Implement if needed

        else:
            logger.error("Either --youtube or --srt must be provided.")
            return

        if video_info.transcription_orig:
            logger.info("Starting summarization...")
            summary_html = youtube_service.summarize_video(video_info, target_language=language)

            if summary_html:
                logger.info("Summarization complete. Saving to files...")
                youtube_service.save_transcription_and_summary(video_info, summary_html, args.output_dir)
                logger.info("Files saved successfully.")
            else:
                logger.info("Summarization failed.")
        else:
            logger.info("No transcription available to summarize.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        youtube_service.db.close()


if __name__ == "__main__":
    main()