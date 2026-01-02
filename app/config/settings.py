from pydantic_settings import BaseSettings

from pydantic_settings import BaseSettings, SettingsConfigDict # Import SettingsConfigDict for V2 compatibility/clarity
from typing import Optional # Use Optional if you want to allow them to be missing

class Settings(BaseSettings):
    """Global AI + system configuration"""

    stt: str = "assemblyai/universal-streaming:en"
    llm: str = "google/gemini-2.0-flash"
    tts: str = "cartesia"
    qdrant_api_key: Optional[str] = None
    qdrant_url: Optional[str] = None
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None
    google_api_key: Optional[str] = None
    stt_api_key: Optional[str] = None
    tts_provider: Optional[str] = None

    class Config:
        env_file = ".env"
        # Optional: Add extra='ignore' here IF you still get errors from *other* system variables
        # extra = 'ignore'
class BookingSettings(BaseSettings):
    collection_name: str = "appointments"


class HelpSettings(BaseSettings):
    collection_name: str = "help_requests"

class KnowledgeSettings(BaseSettings):
    collection_name:str = "knowledge_base"


settings = Settings()
booking_settings = BookingSettings()
help_settings = HelpSettings()
knowledge_settings = KnowledgeSettings()