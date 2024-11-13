import sqlite3
from typing import Optional

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
