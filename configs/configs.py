from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Config(BaseSettings):
    # OpenAI settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-3.5-turbo", env='MODEL_NAME')
    temperature: float = Field(default=0.7, env='TEMPERATURE')
    max_tokens: int = Field(default=1000, env='MAX_TOKENS')
    
    # Optional additional settings
    debug_mode: bool = Field(default=False, env='DEBUG_MODE')
    log_level: str = Field(default="INFO", env='LOG_LEVEL')

    # Tavily API settings
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Config:

    return Config()
