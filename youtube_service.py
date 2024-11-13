import logging
from typing import Optional, List
import os
import yt_dlp
from litellm import completion, completion_cost
from litellm.types.utils import ModelResponse

from transcription_service import TranscriptionService
from DeepInfraAudioClient import Segment
from SRTFile import SRTFile, TimeCode, SubtitleEntry
from VideoInfo import VideoInfo
from database import TranscriptionDatabase

logger = logging.getLogger(__name__)

def extract_prompt(content: str):
    """
        # Extract the prompt from the content that starts with:
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
        start_time = TimeCode(
            hours=int(segment.start) // 3600,
            minutes=(int(segment.start) % 3600) // 60,
            seconds=int(segment.start) % 60,
            milliseconds=int((segment.start % 1) * 1000)
        )
        end_time = TimeCode(
            hours=int(segment.end) // 3600,
            minutes=(int(segment.end) % 3600) // 60,
            seconds=int(segment.end) % 60,
            milliseconds=int((segment.end % 1) * 1000)
        )

        srt_file.add_subtitle(idx, start_time, end_time, [segment.text])

    return str(srt_file)


class YouTubeService:
    def __init__(self, deepinfra_api_key: str, gemini_pro_api_key: str):
        self.deepinfra_api_key = deepinfra_api_key
        self.gemini_pro_api_key = gemini_pro_api_key
        self.SUMMARY_MODEL = "gemini/gemini-1.5-flash-002"
        self.db = TranscriptionDatabase()

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
                    title=video.get('fulltitle', video.get('title', '')),
                    description=video.get('description', ''),
                    filename=filename,
                    ext=ext,
                    language=video.get('language', ''),
                    chapters=video.get('chapters', [])
                )
            else:
                raise FileNotFoundError("Audio file not found.")

    def transcribe_audio(self, video_info: VideoInfo) -> VideoInfo:
        """Transcribes the audio file using the DeepInfra API (Whisper large turbo model) via LiteLLM."""
        cached_result = self.db.get_transcription(video_info.id)
        if cached_result:
            logger.info("Using cached transcription")
            transcription, language, segments_json = cached_result
            video_info.transcription_orig = transcription
            video_info.language = language
            video_info.transcription_by_segments = [Segment(**segment) for segment in eval(segments_json)]
            return video_info

        transcription_service = TranscriptionService(api_key=self.deepinfra_api_key)
        initial_prompt = f'Name: {video_info.title}' # self.generate_initial_prompt(video_info)
        # Gemini Pro costs $0.001875 per second
        logger.debug(f"Transcribing audio file: {video_info.filename}")
        response = transcription_service.transcribe_audio_file(
            video_info.filename,
            language=video_info.language,
            initial_prompt=initial_prompt
        )
        logger.info(f"Segments count: {len(response.segments)}")
        logger.info(f"Transcription cost: ${response.inference_status.cost}, Input tokens: {response.inference_status.tokens_input}, Output tokens: {response.inference_status.tokens_generated}, Runtime: {response.inference_status.runtime_ms} ms")
        video_info.transcription_orig = response.text
        video_info.transcription_by_segments = response.segments

        # Cache the transcription and segments
        segments_json = str([segment.dict() for segment in response.segments])
        self.db.save_transcription(video_info.id, video_info.transcription_orig, video_info.language, segments_json)

        return video_info

    def generate_initial_prompt(self, video_info: VideoInfo) -> str:
        """Generates an initial prompt for the Whisper model using Gemini."""
        # https: // cookbook.openai.com / examples / whisper_prompting_guide

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
                model="gemini/gemini-1.5-pro-002",
                api_key=self.gemini_pro_api_key,
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
        """Summarizes the transcription using the Gemini Pro API via LiteLLM."""
        system_prompt = f"""
        You are an AI assistant specialized in summarizing and translating video content.
        """

        user_prompt = f"""
        You are an AI assistant specialized in summarizing and translating video content. Your task is to create a detailed, insightful summary of a video and translate it into a specified target language.

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
{video_info.chapters}
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

5. Present your translated summary in Markdown format, enclosed within <summary> tags. The entire summary should be in the target language.

Here's an example of how your output should be structured (using English as a placeholder):

<summary>
# Video Summary: [Translated Title]

## Main Topics
- Topic 1
- Topic 2
- Topic 3

## Key Insights
- Insight 1
- Insight 2
- Insight 3

## Non-obvious Ideas
- Idea 1
- Idea 2

## Detailed Summary
[Your translated, detailed summary goes here, using appropriate Markdown formatting for headings, bullet points, emphasis, etc.]

</summary>

Remember to focus on clarity, conciseness, and accuracy in your summary and translation. Do not include any text in the original language in your final output.
        """

        summary_response: ModelResponse = completion(
            model=self.SUMMARY_MODEL,
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

    def save_transcription_and_summary(self, video_info: VideoInfo, summary: str) -> None:
        """Saves the transcription and summary to files with the video ID in the filename."""
        transcription_filename = f"{video_info.id}_transcription.txt"
        summary_filename = f"{video_info.id}_summary.md"
        srt_filename = f"{video_info.id}_transcription.srt"

        self.save_to_file(video_info.transcription_orig, transcription_filename)
        self.save_to_file(summary, summary_filename)
        
        if video_info.transcription_by_segments:
            srt_content = _segments_to_srt(video_info.transcription_by_segments)
            self.save_to_file(srt_content, srt_filename)
            logger.info(f"SRT transcription saved to {srt_filename}")

        logger.info(f"Transcription saved to {transcription_filename}")
        logger.info(f"Summary saved to {summary_filename}")
