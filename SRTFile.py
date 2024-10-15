from dataclasses import dataclass, field
from typing import List
import re


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
        match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", time_string)
        if not match:
            raise ValueError(f"Invalid time format: {time_string}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))


@dataclass
class SubtitleEntry:
    """
    Represents a single subtitle entry in an SRT file.
    """
    index: int # From 1
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
        subtitle = SubtitleEntry(index, start_time, end_time, text)
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
