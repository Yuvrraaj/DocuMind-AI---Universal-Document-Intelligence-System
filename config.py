# config.py - Updated for better rate limiting
import os

# Set your Gemini API key here
GEMINI_API_KEY = ""

# Set environment variable
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# API Configuration - Optimized for free tier
API_RATE_LIMIT = 12       # requests per minute (stay below 15 RPM for flash)
CHUNK_SIZE = 800          # reduced chunk size to lower token usage
CHUNK_OVERLAP = 150       # overlap size for text splitter
MAX_CONTEXT_DOCS = 3      # maximum document chunks to add to context
MAX_RETRIES = 3           # max retries for API calls
RETRY_BASE_DELAY = 30     # base delay in seconds for exponential backoff
CACHE_DIR = "./api_cache" # directory for caching the API responses
