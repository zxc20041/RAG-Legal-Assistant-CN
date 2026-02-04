import os

# ================= 统一配置中心 =================
# 支持环境变量覆盖，便于部署与分发

# --- 云雾 API 配置 ---
YUNWU_API_KEY = os.getenv("YUNWU_API_KEY", "YOUR-YUNWU-API-KEY")
YUNWU_BASE_URL = os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-4o-mini")

# --- 腾讯云配置 ---
TENCENT_SECRET_ID = os.getenv("TENCENT_SECRET_ID", "YOUR-TENCENT-SECRET-ID")
TENCENT_SECRET_KEY = os.getenv("TENCENT_SECRET_KEY", "YOUR-TENCENT-SECRET-KEY")

# --- Flask 应用配置 ---
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "legal_assistant_secret_key233")

# 文件上传配置
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "mp3", "wav", "m4a"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB
MAX_FILES_COUNT = int(os.getenv("MAX_FILES_COUNT", "5"))

# 会话存储配置
SESSION_TYPE = os.getenv("SESSION_TYPE", "filesystem")
SESSION_FILE_DIR = os.getenv("SESSION_FILE_DIR", "./flask_session")
SESSION_PERMANENT = os.getenv("SESSION_PERMANENT", "False").lower() in {"1", "true", "yes", "on"}
SESSION_USE_SIGNER = os.getenv("SESSION_USE_SIGNER", "True").lower() in {"1", "true", "yes", "on"}
SESSION_KEY_PREFIX = os.getenv("SESSION_KEY_PREFIX", "legal_assistant:")

# SQLite 数据库配置
SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///conversations.db")
SQLALCHEMY_TRACK_MODIFICATIONS = os.getenv("SQLALCHEMY_TRACK_MODIFICATIONS", "False").lower() in {"1", "true", "yes", "on"}

# 后端API服务的完整地址
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/predict")

# RAG 调试开关（环境变量 RAG_DEBUG=1/true/on）
RAG_DEBUG = os.getenv("RAG_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
