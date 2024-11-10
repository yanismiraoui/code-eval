import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face token configuration
HF_TOKEN = os.getenv('HF_TOKEN')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model configurations
MODEL_PATHS = {
    'deepseek': 'deepseek-ai/deepseek-coder-1.3b-base',
    'codegemma': 'google/codegemma-2b'
}

# Default configurations
DEFAULT_MAX_LENGTH = 1024
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_PROBLEMS = None
DEFAULT_NUM_SAMPLES = 1