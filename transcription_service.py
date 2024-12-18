# transcription_service.py

import logging
from typing import List, Optional

import backoff

from DeepInfraAudioClient import DeepInfraAudioClient, AutomaticSpeechRecognitionOut
from audio_splitter import AudioChunksSplitter

logger = logging.getLogger(__name__)

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
    def transcribe_chunk(self, audio_chunk, language: Optional[str] = None, initial_prompt: Optional[str] = None) -> AutomaticSpeechRecognitionOut:
        """Transcribe a single audio chunk with retries"""
        return self.client.transcribe(audio_chunk, language=language, initial_prompt=initial_prompt)

    def transcribe_audio_file_splitted(self, audio_file: str, language: Optional[str] = None, initial_prompt: Optional[str] = None) -> AutomaticSpeechRecognitionOut:
        """Split audio file into chunks and transcribe each chunk"""
        # Split audio into chunks
        chunks = AudioChunksSplitter.load_and_chunk_audio(audio_file)
        logger.info(f"Split audio into {len(chunks)} chunks")

        # Transcribe each chunk
        results: List[AutomaticSpeechRecognitionOut] = []
        total_cost = 0
        total_runtime = 0
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
            try:
                chunk_result = self.transcribe_chunk(chunk, language, initial_prompt)
                results.append(chunk_result)
                
                # Accumulate metrics
                total_cost += chunk_result.inference_status.cost
                total_runtime += chunk_result.inference_status.runtime_ms
                
                logger.info(f"Chunk {i+1} transcribed successfully")
                logger.debug(f"Chunk {i+1} cost: ${chunk_result.inference_status.cost:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i+1}: {str(e)}")
                raise

        # Combine results
        combined_result = self._combine_results(results)
        combined_result.inference_status.cost = total_cost
        combined_result.inference_status.runtime_ms = total_runtime
        
        return combined_result

    def _combine_results(self, results: List[AutomaticSpeechRecognitionOut]) -> AutomaticSpeechRecognitionOut:
        """Combine multiple transcription results into one"""
        if not results:
            raise ValueError("No results to combine")
            
        # Use the first result as base
        combined = results[0]
        
        # Combine text and segments from subsequent results
        for result in results[1:]:
            combined.text += " " + result.text
            if result.segments:
                # Adjust timestamps for segments
                last_end = combined.segments[-1].end if combined.segments else 0
                for segment in result.segments:
                    segment.start += last_end
                    segment.end += last_end
                    combined.segments.extend([segment])
                    
        return combined


    def transcribe_audio(self, audio_file: str, language: Optional[str] = None, initial_prompt: Optional[str] = None) -> AutomaticSpeechRecognitionOut:
        """Transcribes the audio file using the DeepInfra API (Whisper large turbo model) via LiteLLM."""
        response = self.client.transcribe(audio_file, language=language, initial_prompt=initial_prompt)
        return response