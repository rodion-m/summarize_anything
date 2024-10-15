import os
from typing import Optional, List, Union, Dict
from enum import Enum
import requests
from pydantic import BaseModel, Field, HttpUrl

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

class DeepInfraAudioClientError(Exception):
    """Custom exception for DeepInfraAudioClient errors"""

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
            raise DeepInfraAudioClientError(f"Error during API request: {str(e)}") from e
        except Exception as e:
            raise DeepInfraAudioClientError(f"Unexpected error: {str(e)}") from e

    def _prepare_audio_file(self, audio_file: Union[str, bytes]) -> Dict[str, Union[str, bytes]]:
        if isinstance(audio_file, str):
            if not os.path.isfile(audio_file):
                raise DeepInfraAudioClientError(f"Audio file not found: {audio_file}")
            return {"audio": open(audio_file, "rb")}
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
