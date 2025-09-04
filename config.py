"""
Configuration settings for the Value Comparison Tool.
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the application."""
    
    # Application settings
    APP_NAME = "Value Comparison Tool"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # File settings
    MAX_FILE_SIZE_MB = 100
    SUPPORTED_FILE_TYPES = ['.xlsx', '.xls']
    TEMP_DIR = "temp"
    
    # Matching settings
    DEFAULT_SIMILARITY_THRESHOLD = 75.0
    MIN_SIMILARITY_THRESHOLD = 50.0
    MAX_SIMILARITY_THRESHOLD = 100.0
    DEFAULT_MAX_MATCHES = 5
    MAX_MAX_MATCHES = 10
    
    # Performance settings
    CACHE_SIZE_MB = 500
    BATCH_SIZE = 100
    MAX_SAMPLE_SIZE = 10000
    EMBEDDING_SAMPLE_SIZE = 5000
    
    # UI settings
    MAX_DISPLAY_ROWS = 1000
    MAX_TEXT_LENGTH = 100
    CHART_HEIGHT = 400
    
    # Database settings
    PRESET_DB_FILE = "Preset 25.xlsx"
    
    # Matching algorithm weights
    ALGORITHM_WEIGHTS = {
        'exact_match': 1.0,
        'fuzzy_ratio': 0.9,
        'fuzzy_partial': 0.8,
        'fuzzy_token_sort': 0.85,
        'fuzzy_token_set': 0.8,
        'jaro_winkler': 0.75,
        'levenshtein': 0.7,
        'pattern_match': 0.6,
        'semantic_match': 0.65
    }
    
    # Unit conversion patterns
    UNIT_PATTERNS = {
        'length': {
            'mm': 1.0,
            'cm': 10.0,
            'm': 1000.0,
            'in': 25.4,
            'inch': 25.4,
            'ft': 304.8
        },
        'weight': {
            'g': 1.0,
            'kg': 1000.0,
            'lb': 453.592,
            'oz': 28.3495
        },
        'voltage': {
            'v': 1.0,
            'mv': 0.001,
            'kv': 1000.0
        },
        'current': {
            'a': 1.0,
            'ma': 0.001,
            'ua': 0.000001
        },
        'power': {
            'w': 1.0,
            'mw': 0.001,
            'kw': 1000.0
        },
        'frequency': {
            'hz': 1.0,
            'khz': 1000.0,
            'mhz': 1000000.0,
            'ghz': 1000000000.0
        },
        'temperature': {
            'c': 1.0,
            'f': 0.555556,  # Conversion factor
            'k': 1.0
        }
    }
    
    # Common synonyms and abbreviations
    SYNONYMS = {
        'voltage': ['v', 'volt', 'volts'],
        'current': ['a', 'amp', 'amps', 'ampere', 'amperes'],
        'power': ['w', 'watt', 'watts'],
        'frequency': ['hz', 'hertz'],
        'resistance': ['ohm', 'ohms', 'r'],
        'capacitance': ['f', 'farad', 'farads'],
        'inductance': ['h', 'henry', 'henries'],
        'temperature': ['temp', '°c', '°f', 'celsius', 'fahrenheit'],
        'length': ['l', 'len', 'long'],
        'width': ['w', 'wide'],
        'height': ['h', 'high', 'tall'],
        'diameter': ['dia', 'd'],
        'thickness': ['thick', 't']
    }
    
    # Common abbreviations
    ABBREVIATIONS = {
        'max': 'maximum',
        'min': 'minimum',
        'nom': 'nominal',
        'typ': 'typical',
        'std': 'standard',
        'stdby': 'standby',
        'pwr': 'power',
        'gnd': 'ground',
        'ref': 'reference',
        'temp': 'temperature',
        'volt': 'voltage',
        'amp': 'current',
        'watt': 'power',
        'ohm': 'resistance',
        'farad': 'capacitance',
        'henry': 'inductance'
    }
    
    # Status colors for UI
    STATUS_COLORS = {
        'Exact Match': '#28a745',
        'Similar Match': '#ffc107', 
        'Not Found': '#dc3545',
        'Processing': '#17a2b8',
        'Error': '#dc3545'
    }
    
    # Error messages
    ERROR_MESSAGES = {
        'file_too_large': f"File size exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB",
        'invalid_file_type': f"Invalid file type. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}",
        'missing_columns': "Required columns are missing from the uploaded file",
        'empty_file': "The uploaded file is empty",
        'processing_error': "An error occurred while processing the file",
        'database_not_loaded': "Please load a reference database first",
        'no_input_data': "Please upload input data to compare",
        'no_results': "No comparison results available"
    }
    
    # Success messages
    SUCCESS_MESSAGES = {
        'database_loaded': "Reference database loaded successfully",
        'input_loaded': "Input data loaded successfully",
        'comparison_complete': "Comparison completed successfully",
        'export_complete': "Results exported successfully",
        'settings_saved': "Settings saved successfully"
    }
    
    # Warning messages
    WARNING_MESSAGES = {
        'large_file': "Large file detected. Processing may take longer",
        'many_matches': "Many potential matches found. Consider adjusting similarity threshold",
        'low_similarity': "Low similarity scores detected. Results may not be accurate",
        'model_loading': "Loading AI model for semantic matching. This may take a moment"
    }

# Create global config instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update configuration settings."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")

def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific configuration setting."""
    return getattr(config, key, default)