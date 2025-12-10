from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global AI + system configuration"""

    stt: str = "assemblyai/universal-streaming:en"
    llm: str = "google/gemini-2.0-flash"
    tts: str = "cartesia"

    class Config:
        env_file = ".env"


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