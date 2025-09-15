# Swecha API Configuration
# This file contains configuration settings for the Swecha Corpus API integration

# API Base URL
SWECHA_API_BASE_URL = "https://api.corpus.swecha.org"

# API Endpoints
API_ENDPOINTS = {
    "health": "/health",
    "content_search": "/content/search",
    "files_search": "/files/search",
    "general_search": "/search",
    "content_upload": "/content/upload",
    "files_upload": "/files/upload",
    "content": "/content",
    "categories": "/categories",
    "content_types": "/content-types",
    "stats": "/stats"
}

# API Request Settings
API_TIMEOUT = 30  # seconds
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1  # seconds

# Authentication Settings
API_AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTY3OTc2NTAsInN1YiI6IjY4NjUzMjkwLTdiYzEtNGYyZi1hMjI5LTBhNDQ4MGZjZWE0NyJ9.gmxj38_bp7pkv7P5ZA3UMOVG94-628tqsoYROtknopI"  # Your Swecha API authentication token
API_AUTH_HEADER = "Authorization"
API_AUTH_TYPE = "Bearer"  # or "Token" depending on your API

# Search Settings
DEFAULT_SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 100

# Upload Settings
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']
SUPPORTED_TEXT_FORMATS = ['.txt', '.pdf', '.doc', '.docx', '.rtf']

# Content Categories (mapping to your app's categories)
CONTENT_CATEGORIES = {
    "monuments": {
        "id": "monuments",
        "name": "స్మారకాలు",
        "description": "Historical monuments and landmarks",
        "tags": ["monuments", "historical", "landmarks", "architecture"]
    },
    "culture": {
        "id": "culture", 
        "name": "సంస్కృతి",
        "description": "Cultural heritage and traditions",
        "tags": ["culture", "heritage", "traditions", "telugu"]
    },
    "traditions": {
        "id": "traditions",
        "name": "సంప్రదాయాలు", 
        "description": "Traditional practices and customs",
        "tags": ["traditions", "customs", "practices", "telugu"]
    },
    "folktales": {
        "id": "folktales",
        "name": "జానపద కథలు",
        "description": "Folk tales and stories",
        "tags": ["folktales", "stories", "narratives", "telugu"]
    }
}

# Content Types
CONTENT_TYPES = {
    "images": {
        "id": "images",
        "name": "చిత్రాలు",
        "description": "Images and photographs",
        "extensions": SUPPORTED_IMAGE_FORMATS
    },
    "videos": {
        "id": "videos", 
        "name": "వీడియోలు",
        "description": "Video content",
        "extensions": SUPPORTED_VIDEO_FORMATS
    },
    "texts": {
        "id": "texts",
        "name": "టెక్స్ట్ ఫైల్స్",
        "description": "Text documents and files",
        "extensions": SUPPORTED_TEXT_FORMATS
    }
}

# Default Metadata
DEFAULT_METADATA = {
    "language": "te",  # Telugu language code
    "country": "IN",   # India
    "region": "TG",    # Telangana
    "tags": ["telugu", "cultural_heritage", "india", "telangana"],
    "license": "CC-BY-SA",  # Creative Commons license
    "source": "telugu_cultural_heritage_app"
}

# Error Messages (in Telugu)
ERROR_MESSAGES = {
    "api_unavailable": "Swecha API అందుబాటులో లేదు",
    "upload_failed": "అప్‌లోడ్ విఫలం",
    "search_failed": "శోధన విఫలం",
    "network_error": "నెట్‌వర్క్ లోపం",
    "file_too_large": "ఫైల్ చాలా పెద్దది",
    "unsupported_format": "ఆమోదించబడని ఫైల్ ఫార్మాట్",
    "authentication_required": "ప్రమాణీకరణ అవసరం"
}

# Success Messages (in Telugu)
SUCCESS_MESSAGES = {
    "upload_success": "ఫైల్ విజయవంతంగా అప్‌లోడ్ చేయబడింది",
    "search_success": "శోధన విజయవంతం",
    "api_connected": "Swecha API కనెక్ట్ అయింది"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "swecha_api.log"
}
