import os

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


class Settings:
    GENAIHUB_API_KEY: SecretStr = SecretStr(os.environ["GENAIHUB_API_KEY"])
    GENAIHUB_API_VERSION: str = os.environ["GENAIHUB_API_VERSION"]
    GENAIHUB_CERT_PATH: str = os.environ["GENAIHUB_CERT_PATH"]


settings = Settings()
