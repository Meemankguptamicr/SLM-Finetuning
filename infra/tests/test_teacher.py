from dotenv import load_dotenv
from os import getenv
from utils import do_test_openai_endpoint

load_dotenv(".env")
load_dotenv(".env.state")

def test_teacher():
    endpoint = getenv("COMPLETION_AZURE_OPENAI_ENDPOINT")
    key = getenv("COMPLETION_AZURE_OPENAI_API_KEY")
    model = getenv("COMPLETION_AZURE_OPENAI_DEPLOYMENT")
    version = getenv("COMPLETION_AZURE_OPENAI_API_VERSION")
    do_test_openai_endpoint(endpoint, key, model,version)
