"""
Configuration module for CarePal Support Assistant (Phase 1).
Contains project paths, target URLs, scraping rules, chunking parameters,
and vector database settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Base Paths & Directories
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

RAW_PAGES_FILE = DATA_DIR / "raw_pages.json"
PROCESSED_CHUNKS_FILE = DATA_DIR / "processed_chunks.json"
VECTOR_INDEX_FILE = VECTOR_DB_DIR / "index.faiss"
METADATA_FILE = VECTOR_DB_DIR / "metadata.json"
CHUNKS_FILE = VECTOR_DB_DIR / "chunks.json"
SQLITE_DB_PATH = DATA_DIR / "carepal_ops.db"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Website & Target URLs
# ------------------------------------------------------------------------------
BASE_URL = "https://www.thecarepal.com"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

# Core pages mapping (essential for customer support, service inquiries & policies)
CORE_TARGET_PAGES = [
    # Home & Company
    {
        "url": f"{BASE_URL}/",
        "category": "company",
        "label": "Home",
        "description": "TheCarePal homepage, mission, core care services overview and family testimonials."
    },
    {
        "url": f"{BASE_URL}/contact",
        "category": "support",
        "label": "Contact Us",
        "description": "Contact channels, phone numbers, email, 24/7 care support and consultation booking."
    },
    {
        "url": f"{BASE_URL}/careers",
        "category": "company",
        "label": "Careers",
        "description": "Job openings, caregiver opportunities and recruitment at TheCarePal."
    },
    {
        "url": f"{BASE_URL}/lisa",
        "category": "product",
        "label": "Lisa AI Companion",
        "description": "Information about Lisa, TheCarePal's AI healthcare assistant and companion."
    },

    # Core Care Services (with integrated FAQs & service terms)
    {
        "url": f"{BASE_URL}/elderly",
        "category": "service",
        "label": "Elderly Care",
        "description": "Professional in-home elder care, mobility assistance, medication management and FAQs."
    },
    {
        "url": f"{BASE_URL}/dementia",
        "category": "service",
        "label": "Dementia Care",
        "description": "Specialized memory support, patient safety, trained dementia caregivers and FAQs."
    },
    {
        "url": f"{BASE_URL}/alzeihmer",
        "category": "service",
        "label": "Alzheimer's Care",
        "description": "Specialized Alzheimer's patient care, emotional comfort, routine management and FAQs."
    },
    {
        "url": f"{BASE_URL}/livein-care",
        "category": "service",
        "label": "24/7 Live-In Care",
        "description": "Round-the-clock live-in caregiver support at home, caregiver shifts and FAQs."
    },
    {
        "url": f"{BASE_URL}/babycare",
        "category": "service",
        "label": "Baby Care",
        "description": "Newborn care, infant feeding, developmental tracking, post-natal mother support and FAQs."
    },
    {
        "url": f"{BASE_URL}/autism",
        "category": "service",
        "label": "Autism Care",
        "description": "Specialized pediatric and youth support for autism spectrum, behavioral therapy and FAQs."
    },
    {
        "url": f"{BASE_URL}/therapy",
        "category": "service",
        "label": "Mental Therapy & Counseling",
        "description": "Licensed counselors, clinical psychologists, depression, anxiety and family counseling."
    },
    {
        "url": f"{BASE_URL}/doctor",
        "category": "service",
        "label": "Doctor Consultation",
        "description": "Online doctor consultations, physician appointments, clinical reviews and FAQs."
    },
    {
        "url": f"{BASE_URL}/emergency",
        "category": "service",
        "label": "Emergency Care Support",
        "description": "Immediate urgent assistance, medical emergency protocols and ambulance support."
    },
    {
        "url": f"{BASE_URL}/pharmaceutical",
        "category": "service",
        "label": "Pharmaceutical Delivery",
        "description": "Genuine prescription medicines, doorstep delivery, refill management and storage."
    },
    {
        "url": f"{BASE_URL}/services",
        "category": "service",
        "label": "All Services Overview",
        "description": "Directory of all home care services provided across India."
    },
    {
        "url": f"{BASE_URL}/consultation",
        "category": "service",
        "label": "Consultation Booking",
        "description": "Free care assessment booking with clinical experts."
    },
    {
        "url": f"{BASE_URL}/products",
        "category": "product",
        "label": "Care Products",
        "description": "Medical aids, mobility devices and elder-safe home equipment."
    },

    # Legal & Policies (Critical for refund, pricing, terms & cancellation queries)
    {
        "url": f"{BASE_URL}/refund",
        "category": "policy",
        "label": "Refund Policy",
        "description": "Refund terms, cancellation rules, conditions for refunds, payment disputes and contact info."
    },
    {
        "url": f"{BASE_URL}/terms",
        "category": "policy",
        "label": "Terms & Conditions",
        "description": "Conditions of use, service agreement, user accounts, liability limits and disputes."
    },
    {
        "url": f"{BASE_URL}/privacy",
        "category": "policy",
        "label": "Privacy Policy",
        "description": "User data collection, health data handling, consent, cookies and data protection rights."
    },

    # Blog Hub
    {
        "url": f"{BASE_URL}/blogs",
        "category": "blog",
        "label": "Health Blogs & Resources",
        "description": "Caregiving tips, medication management guides and senior wellness articles."
    }
]

# ------------------------------------------------------------------------------
# 3. Web Scraper Settings
# ------------------------------------------------------------------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 CarePalBot/1.0"
)
USE_HEADLESS_BROWSER = os.getenv("USE_HEADLESS_BROWSER", "false").lower() in ("true", "1", "yes")

# Boilerplate tags and CSS classes to systematically decompose from the DOM
BOILERPLATE_TAGS = ["nav", "footer", "script", "style", "noscript", "svg", "iframe", "canvas"]
BOILERPLATE_CLASSES = [
    "navbar-container",
    "navbar",
    "footer",
    "floating-chat-btn",
    "mobile-drawer",
    "cookie-banner",
    "chat-widget",
    "modal-backdrop"
]

# ------------------------------------------------------------------------------
# 4. Text Chunking Settings
# ------------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

# ------------------------------------------------------------------------------
# 5. Embedding & Vector Database Settings
# ------------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
NORMALIZE_EMBEDDINGS = True  # Cosine similarity via inner product

# ------------------------------------------------------------------------------
# 6. RAG Generation & Guardrail Settings (Phase 2)
# ------------------------------------------------------------------------------
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "8"))
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

