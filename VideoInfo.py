from typing import Optional, List

from DeepInfraAudioClient import Segment


class VideoInfo:
    def __init__(self, yt_dlp_info: dict, id: str, title: str, description: str, filename: str, ext: str, language: Optional[str],
                 chapters: List[dict]):
        self.yt_dlp_info: dict = yt_dlp_info
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
