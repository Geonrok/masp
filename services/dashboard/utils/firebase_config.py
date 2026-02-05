"""Firebase configuration for Streamlit dashboard.

SSOT: All Firebase config values in one place.
Reads from environment variables for security.
"""

import os

# Firebase Web App Configuration (from environment variables)
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", ""),
    "authDomain": os.getenv(
        "FIREBASE_AUTH_DOMAIN", "multi-asset-strategy-platform.firebaseapp.com"
    ),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", "multi-asset-strategy-platform"),
    "storageBucket": os.getenv(
        "FIREBASE_STORAGE_BUCKET", "multi-asset-strategy-platform.firebasestorage.app"
    ),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
    "appId": os.getenv("FIREBASE_APP_ID", ""),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", ""),
    # Required for pyrebase
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", ""),
}


def is_firebase_configured() -> bool:
    """Check if Firebase is properly configured."""
    return bool(FIREBASE_CONFIG.get("apiKey"))
