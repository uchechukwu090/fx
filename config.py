import os
from dataclasses import dataclass, field
from typing import Optional
import logging

@dataclass
class Config:
    """Configuration class for the trading bot application"""

    # Flask Configuration
    FLASK_ENV: str = os.getenv('FLASK_ENV', 'production')
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    PORT: int = int(os.getenv('PORT', 5000))
    HOST: str = os.getenv('HOST', '0.0.0.0')

    # API Configuration - Twelve Data
    TWELVE_DATA_API_KEY: str = os.getenv('TWELVE_DATA_API_KEY', 'fef3c30aa26c4831924fdb142f87550d')
    TWELVE_DATA_BASE_URL: str = os.getenv('TWELVE_DATA_BASE_URL', 'https://api.twelvedata.com/v1')
    API_RATE_LIMIT: int = int(os.getenv('API_RATE_LIMIT', 60))  # requests per minute
    API_TIMEOUT: int = int(os.getenv('API_TIMEOUT', 10))  # seconds

    # Database Configuration
    DATABASE_PATH: str = os.getenv('DATABASE_PATH', '/app/data/trading_bot.db')
    DATABASE_POOL_SIZE: int = int(os.getenv('DATABASE_POOL_SIZE', 5))

    # Trading System Configuration
    RISK_TOLERANCE: float = float(os.getenv('RISK_TOLERANCE', 0.02))
    WAVELET_TYPE: str = os.getenv('WAVELET_TYPE', 'db4')
    PROCESS_NOISE: float = float(os.getenv('PROCESS_NOISE', 1e-5))
    MEASUREMENT_NOISE: float = float(os.getenv('MEASUREMENT_NOISE', 1e-1))

    # Analysis Configuration
    MIN_DATA_POINTS: int = int(os.getenv('MIN_DATA_POINTS', 32))
    DEFAULT_HISTORICAL_DAYS: int = int(os.getenv('DEFAULT_HISTORICAL_DAYS', 100))
    MAX_HISTORICAL_DAYS: int = int(os.getenv('MAX_HISTORICAL_DAYS', 365))

    # Background Tasks Configuration
    UPDATE_INTERVAL: int = int(os.getenv('UPDATE_INTERVAL', 300))  # seconds
    BACKGROUND_SYMBOLS: list = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'BTC/USD', 'ETH/USD'])

    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE', None)

    # CORS Configuration
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', '*')
    CORS_METHODS: str = os.getenv('CORS_METHODS', 'GET,POST,PUT,DELETE,OPTIONS')
    CORS_HEADERS: str = os.getenv('CORS_HEADERS', 'Content-Type,Authorization')

    # Performance Configuration
    GUNICORN_WORKERS: int = int(os.getenv('GUNICORN_WORKERS', 2))
    GUNICORN_THREADS: int = int(os.getenv('GUNICORN_THREADS', 4))
    GUNICORN_TIMEOUT: int = int(os.getenv('GUNICORN_TIMEOUT', 120))
    GUNICORN_WORKER_CLASS: str = os.getenv('GUNICORN_WORKER_CLASS', 'sync')

    # Cache Configuration
    CACHE_TIMEOUT: int = int(os.getenv('CACHE_TIMEOUT', 60))  # seconds
    MAX_CACHE_SIZE: int = int(os.getenv('MAX_CACHE_SIZE', 100))

    # Security Configuration
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

    # Render.com specific
    RENDER_EXTERNAL_URL: Optional[str] = os.getenv('RENDER_EXTERNAL_URL')
    RENDER_SERVICE_NAME: Optional[str] = os.getenv('RENDER_SERVICE_NAME')

    def __post_init__(self):
        """Post-initialization validation and setup"""
        os.makedirs(os.path.dirname(self.DATABASE_PATH), exist_ok=True)
        if not self.TWELVE_DATA_API_KEY or self.TWELVE_DATA_API_KEY == 'your-api-key-here':
            logging.warning("Using default/placeholder API key. Please set TWELVE_DATA_API_KEY environment variable.")
        if not 0 < self.RISK_TOLERANCE <= 1:
            raise ValueError("RISK_TOLERANCE must be between 0 and 1")
        valid_wavelets = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'haar', 'bior2.2', 'coif2']
        if self.WAVELET_TYPE not in valid_wavelets:
            logging.warning(f"Wavelet type {self.WAVELET_TYPE} may not be valid. Valid types: {valid_wavelets}")

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        return cls()

    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging_config = {
            'level': log_level,
            'format': self.LOG_FORMAT,
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        if self.LOG_FILE:
            logging_config['filename'] = self.LOG_FILE
            logging_config['filemode'] = 'a'
        logging.basicConfig(**logging_config)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

    def get_database_url(self) -> str:
        """Get database URL for SQLAlchemy if needed"""
        return f"sqlite:///{self.DATABASE_PATH}"

    def get_cors_config(self) -> dict:
        """Get CORS configuration"""
        return {
            'origins': self.CORS_ORIGINS.split(',') if self.CORS_ORIGINS != '*' else '*',
            'methods': self.CORS_METHODS.split(','),
            'allow_headers': self.CORS_HEADERS.split(',')
        }

    def get_twelve_data_headers(self) -> dict:
        """Get headers for Twelve Data API requests"""
        return {
            'Authorization': f'apikey {self.TWELVE_DATA_API_KEY}',
            'Content-Type': 'application/json'
        }

    def get_twelve_data_params(self, **kwargs) -> dict:
        """Get base parameters for Twelve Data API requests"""
        params = {
            'apikey': self.TWELVE_DATA_API_KEY,
            **kwargs
        }
        return params

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.FLASK_ENV == 'production'

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.FLASK_ENV == 'development'

# Global config instance
config = Config.from_env()

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    DATABASE_PATH = ':memory:'
    TWELVE_DATA_API_KEY = 'test-key'

config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

def get_config(env: str = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'production')
    config_class = config_mapping.get(env, ProductionConfig)
    return config_class()
