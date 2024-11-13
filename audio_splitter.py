import logging
import os
import sys
import time
from typing import List
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Configs
load_dotenv()
MAX_CHUNK_SIZE_MB = int(os.getenv("MAX_CHUNK_SIZE_MB", 25))
MAX_CHUNK_DURATION_SEC = int(os.getenv("MAX_CHUNK_DURATION_SEC", 30))
INITIAL_MIN_SILENCE_LEN_MS = int(os.getenv("INITIAL_MIN_SILENCE_LEN_MS", 3000))
MIN_SILENCE_LEN_STEP = int(os.getenv("MIN_SILENCE_LEN_STEP", 100))

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # set logger level

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # set console handler level

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Add formatter to both file and console handlers
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)


class AudioChunksSplitter:
    # Perhaps a better alternative: https://github.com/snakers4/silero-vad
    # Example for silero-vad: https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb#scrollTo=QttWasy5hUd6
    @staticmethod
    def split_large_chunks(chunks: List[AudioSegment], max_duration: int, min_silence_len: int) -> List[AudioSegment]:
        """
        Split large chunks into smaller chunks based on silence until all chunks are less than max_duration
        """
        logging.info(f"Splitting {len(chunks)} chunks with min_silence_len={min_silence_len}ms...")
        if min_silence_len < 0:
            logging.info("We can't split on silence, splitting straight...")
            return AudioChunksSplitter.split_straight(chunks, max_duration)
        elif min_silence_len == 0:
            min_silence_len = 30

        small_chunks = []
        for chunk in chunks:
            if chunk.duration_seconds > max_duration:
                small_chunks.extend(
                    split_on_silence(chunk, min_silence_len, silence_thresh=chunk.dBFS - 14, keep_silence=1000))
            else:
                small_chunks.append(chunk)

        logging.info(f"Split large chunks into {len(small_chunks)} chunks")

        largest_chunk = max(small_chunks, key=lambda x: x.duration_seconds)
        logging.info(f"Largest chunk duration: {largest_chunk.duration_seconds}")

        # check if any chunk has duration greater than max_duration and call split_large_chunks recursively
        if largest_chunk.duration_seconds > max_duration:
            return AudioChunksSplitter.split_large_chunks(small_chunks, max_duration,
                                                          min_silence_len - MIN_SILENCE_LEN_STEP)

        return small_chunks

    @staticmethod
    def load_and_chunk_audio(file_path: str) -> List[AudioSegment]:
        logging.info("Loading audio file...")
        start_time = time.time()
        audio = AudioSegment.from_file(file_path)
        logging.info(f"Audio file loaded in {time.time() - start_time:.2f} seconds")
        logging.info(f"Audio duration: {audio.duration_seconds} seconds")

        logging.info("Splitting audio on silence...")
        start_time = time.time()
        # Alternative: https://librosa.org/doc/0.10.2/generated/librosa.effects.split.html
        chunks = AudioChunksSplitter.split_large_chunks([audio], MAX_CHUNK_DURATION_SEC, INITIAL_MIN_SILENCE_LEN_MS)
        logging.info(f"Audio split into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")

        logging.info("Merging chunks...")
        start_time = time.time()
        merged_chunks = []
        current_chunk = AudioSegment.empty()

        for chunk in chunks:
            if (len(current_chunk.raw_data) + len(chunk.raw_data) <= MAX_CHUNK_SIZE_MB * 1024 * 1024
                    and current_chunk.duration_seconds + chunk.duration_seconds <= MAX_CHUNK_DURATION_SEC):
                current_chunk += chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            merged_chunks.append(current_chunk)

        logging.info(f"Merged into {len(merged_chunks)} chunks in {time.time() - start_time:.2f} seconds")
        return merged_chunks

    @staticmethod
    def split_straight(chunks: List[AudioSegment], max_duration: int) -> List[AudioSegment]:
        """
        Strictly split chunks into smaller chunks based on max_duration
        """
        logging.info(f"Splitting {len(chunks)} chunks straight into {max_duration} seconds...")
        start_time = time.time()
        small_chunks = []
        for chunk in chunks:
            if chunk.duration_seconds > max_duration:
                small_chunks.extend(chunk[:max_duration * 1000].split_to_mono())
            else:
                small_chunks.append(chunk)

        logging.info(f"Split into {len(small_chunks)} chunks in {time.time() - start_time:.2f} seconds")

        return small_chunks

# Speaker diarization and timing out of the box are available in WhisperX: https://github.com/m-bain/whisperX
# LangChain implementation: https://github.com/langchain-ai/langchain/blob/c03899159050d33bbc199e415cf12cb933efd0fb/libs/community/langchain_community/document_loaders/parsers/audio.py#L117