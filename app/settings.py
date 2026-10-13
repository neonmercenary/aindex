from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str
    EMBEDDING_MODEL_NAME: str
    MAX_DOCUMENTS: str

    class Config:
        env_file = ".env"


settings = Settings()