"""Gestione delle variabili d'ambiente tramite Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    chroma_path: str = "./chroma_db"
    collection_name: str = "mediterranean_recipes"
    seed_path: str = "./data/seed_recipes.json"
    llm_temperature: float = 0.2
    retrieval_top_k: int = 25

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()