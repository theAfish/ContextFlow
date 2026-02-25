from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    app_name: str = "ContextFlow"
    app_env: str = Field(default="dev", pattern=r"^(dev|test|prod)$")
    max_context_tokens: int = 16_000


settings = Settings()
