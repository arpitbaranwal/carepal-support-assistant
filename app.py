"""
CarePal Support Assistant - Mobile Chat Widget (Drift / LeadBot Style)
Phase 2 Implementation:
- TheCarePal Official Brand Palette: Dark Slate-Navy (#181F2B), Card (#202836),
  Periwinkle-Lavender (#9F8FEF), and Teal-Mint (#00D09C / #10B981)
- Seamless bottom chat input blending into card with no mismatched black box
- Centered, sleek mobile card widget (max-width: 440px)
- Solid header banner with bot avatar & glowing teal online status
- Scrollable message area inside the widget card (no full-screen scrolling)
- Question asking input (st.chat_input) cleanly integrated inside the widget card
- Ops Admin toggle positioned at the top-right corner of the whole page
- Fast retrieval with pre-cached embedding model & real-time streaming
- Language-matched fallback for English, Hindi, and Hinglish queries
- Compact inline lead capture card (Name + 10-digit Phone) for out-of-scope queries
- Clickable citation pills below bot answers
"""

import sys
import re
import html
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db

# ------------------------------------------------------------------------------
# 1. Page Configuration & Pre-warming Cache
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CarePal | Home Care Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.rag_engine import rag_engine

# Pre-warm embedding model & vector store to eliminate cold start
if not rag_engine.vector_store_loaded:
    rag_engine.vector_store_loaded = rag_engine.vector_store.load()


def render_html(html_str: str) -> None:
    """
    Renders HTML safely by collapsing all inter-tag whitespace and newlines.
    This eliminates any blank lines + 4-space indentation, preventing CommonMark
    from ever interpreting closing tags (like </div>) as indented code blocks.
    """
    compact_html = re.sub(r">\s+<", "><", re.sub(r"\s+", " ", html_str)).strip()
    st.markdown(compact_html, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. Custom CSS (TheCarePal Brand Palette: Dark Slate, Periwinkle & Teal)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* 1. Global Dark Slate Theme (#181F2B & #202836) */
    :root, [data-theme="dark"], [data-theme="light"], .stApp {
        --background-color: #181F2B !important;
        --secondary-background-color: #202836 !important;
        --text-color: #FFFFFF !important;
        --primary-color: #9F8FEF !important;
        color-scheme: dark !important;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
        background-color: #181F2B !important;
        background: #181F2B !important;
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Completely hide Streamlit Header, Footer, Sidebar & Toolbar */
    header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"], #MainMenu {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Page container spacing */
    .main .block-container {
        padding: 0.75rem 1.5rem 2rem 1.5rem !important;
        max-width: 1280px !important;
        margin: 0 auto !important;
    }

    /* 2. Top-Right Navigation Bar */
    .top-nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        padding: 0 0.25rem;
    }

    .brand-title {
        font-weight: 700;
        font-size: 16px;
        color: #9F8FEF;
        letter-spacing: 0.3px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 12.5px;
        font-weight: 400;
    }

    /* Top-Right Ops Admin Pill Button */
    div.st-key-btn_top_admin button,
    div.st-key-btn_top_chat button {
        background-color: #202836 !important;
        background: #202836 !important;
        color: #C4B5FD !important;
        border: 1.5px solid #9F8FEF !important;
        border-radius: 999px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 5px 16px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
        float: right !important;
    }

    div.st-key-btn_top_admin button:hover,
    div.st-key-btn_top_chat button:hover {
        background-color: #2E3A50 !important;
        background: #2E3A50 !important;
        border-color: #B4A5FF !important;
        color: #FFFFFF !important;
    }

    /* 3. Outer Chat Card Container (#202836) */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-header) {
        background-color: #202836 !important;
        background: #202836 !important;
        border-radius: 20px !important;
        border: 1px solid #2E3A4E !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45) !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 auto !important;
        max-width: 440px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-header) > div {
        background-color: #202836 !important;
        background: #202836 !important;
        gap: 0 !important;
        padding: 0 !important;
    }

    /* 4. Slate & Periwinkle Header Banner with Teal Status */
    .chat-header {
        background: linear-gradient(135deg, #2A3548 0%, #202836 100%) !important;
        padding: 14px 18px;
        border-bottom: 1.5px solid #2E3A4E !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #FFFFFF !important;
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .bot-avatar-circle {
        width: 38px;
        height: 38px;
        background-color: #9F8FEF !important;
        background: #9F8FEF !important;
        color: #181F2B !important;
        border: 2px solid #C4B5FD !important;
        border-radius: 50% !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
        box-shadow: 0 2px 8px rgba(159, 143, 239, 0.35);
        flex-shrink: 0;
    }

    .header-title-box {
        display: flex;
        flex-direction: column;
    }

    .header-title {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF !important;
        line-height: 1.2;
    }

    .header-status {
        font-size: 12px;
        color: #2DD4BF !important;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 2px;
        font-weight: 500;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #00D09C !important;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #00D09C;
    }

    /* 5. Inner Scrollable Message Area */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-header) div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        background-color: #202836 !important;
        background: #202836 !important;
        padding: 12px 14px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-header) div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #202836 !important;
        background: #202836 !important;
    }

    /* Bot Message (Left) */
    .bot-msg-wrapper {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 12px;
    }

    .bot-icon-small {
        width: 28px;
        height: 28px;
        background-color: #9F8FEF !important;
        background: #9F8FEF !important;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: #181F2B !important;
        flex-shrink: 0;
        margin-top: 14px;
        box-shadow: 0 2px 6px rgba(159, 143, 239, 0.25);
    }

    .bot-content-col {
        display: flex;
        flex-direction: column;
        max-width: 84%;
    }

    .bot-author-label {
        font-size: 11px;
        font-weight: 600;
        color: #94A3B8 !important;
        margin-left: 4px;
        margin-bottom: 3px;
    }

    .bot-bubble {
        background-color: #283243 !important;
        background: #283243 !important;
        color: #F1F5F9 !important;
        border: 1px solid #374458 !important;
        border-radius: 18px 18px 18px 6px;
        padding: 11px 15px;
        font-size: 13.5px;
        line-height: 1.5;
        word-wrap: break-word;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }

    .bot-bubble b, .bot-bubble strong {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    .bot-bubble .bot-heading {
        color: #C4B5FD !important;
        font-size: 13.5px !important;
        font-weight: 700 !important;
        margin: 6px 0 3px 0 !important;
        line-height: 1.4 !important;
        display: block !important;
        border-bottom: 1px solid rgba(159, 143, 239, 0.25) !important;
        padding-bottom: 3px !important;
    }

    .bot-bubble .bot-line {
        margin: 2px 0 !important;
        line-height: 1.5 !important;
    }

    .bot-bubble .bot-bullet {
        display: flex !important;
        align-items: flex-start !important;
        gap: 7px !important;
        margin: 3px 0 !important;
        line-height: 1.45 !important;
    }

    .bot-bubble .bot-bullet-dot {
        color: #9F8FEF !important;
        font-size: 11px !important;
        margin-top: 2px !important;
        flex-shrink: 0 !important;
        line-height: 1 !important;
    }

    .bot-bubble .bot-spacer {
        height: 6px !important;
    }

    /* User Message (Right - Periwinkle Gradient) */
    .user-msg-wrapper {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 12px;
    }

    .user-bubble {
        background: linear-gradient(135deg, #9F8FEF 0%, #8270DB 100%) !important;
        color: #FFFFFF !important;
        border-radius: 20px 20px 6px 20px;
        padding: 10px 16px;
        font-size: 13.5px;
        line-height: 1.5;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 3px 10px rgba(159, 143, 239, 0.35);
    }

    /* Citation Badges below Bot Bubble */
    .citations-row {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: 6px;
        margin-left: 4px;
    }

    .citation-pill {
        display: inline-flex;
        align-items: center;
        background-color: #1E2736 !important;
        background: #1E2736 !important;
        color: #C4B5FD !important;
        border: 1px solid #3E4E68 !important;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 11.5px;
        font-weight: 500;
        text-decoration: none !important;
        transition: all 0.15s ease;
    }

    .citation-pill:hover {
        background-color: #2A3548 !important;
        background: #2A3548 !important;
        border-color: #9F8FEF !important;
        color: #FFFFFF !important;
    }

    /* 6. Quick Starter Buttons - Dark Slate with Periwinkle Border */
    div[data-testid="stHorizontalBlock"] button,
    button[data-testid="stBaseButton-secondary"],
    button[kind="secondary"],
    .stButton > button {
        background-color: #202836 !important;
        background: #202836 !important;
        color: #C4B5FD !important;
        border: 1.5px solid #9F8FEF !important;
        border-radius: 999px !important;
        font-size: 11.5px !important;
        font-weight: 500 !important;
        padding: 4px 8px !important;
        margin: 2px 0 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    div[data-testid="stHorizontalBlock"] button:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[kind="secondary"]:hover,
    .stButton > button:hover {
        background-color: #2E3A50 !important;
        background: #2E3A50 !important;
        border-color: #B4A5FF !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stHorizontalBlock"] button p,
    button[data-testid="stBaseButton-secondary"] p,
    .stButton > button p,
    div[data-testid="stHorizontalBlock"] button span,
    button[data-testid="stBaseButton-secondary"] span,
    .stButton > button span {
        color: #C4B5FD !important;
        font-weight: 500 !important;
    }

    div[data-testid="stHorizontalBlock"] button:hover p,
    div[data-testid="stHorizontalBlock"] button:hover span {
        color: #FFFFFF !important;
    }

    /* 7. Inline Lead Capture Card inside Card */
    div[data-testid="stForm"] {
        background-color: #1A2230 !important;
        background: #1A2230 !important;
        border: 1.5px solid #334155 !important;
        border-radius: 14px !important;
        padding: 10px 14px !important;
        margin: 6px 0 10px 32px !important;
        max-width: 86% !important;
    }

    .inline-lead-title {
        font-size: 12.5px;
        font-weight: 600;
        color: #9F8FEF !important;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    div[data-testid="stForm"] div[data-testid="stTextInput"] input {
        background-color: #202836 !important;
        background: #202836 !important;
        border: 1px solid #3B4860 !important;
        border-radius: 8px !important;
        font-size: 12.5px !important;
        color: #FFFFFF !important;
        padding: 6px 10px !important;
    }

    div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
        border-color: #9F8FEF !important;
        box-shadow: 0 0 0 2px rgba(159, 143, 239, 0.25) !important;
    }

    /* Periwinkle submit button in lead form */
    div[data-testid="stForm"] button[kind="primary"],
    div[data-testid="stForm"] button[type="submit"] {
        background-color: #9F8FEF !important;
        background: #9F8FEF !important;
        color: #181F2B !important;
        border-radius: 999px !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 12.5px !important;
        padding: 5px 14px !important;
        width: 100% !important;
        box-shadow: 0 2px 8px rgba(159, 143, 239, 0.35) !important;
        margin-top: 4px !important;
    }

    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: #B4A5FF !important;
    }

    .lead-success-banner {
        background-color: #064E3B !important;
        background: #064E3B !important;
        border: 1px solid #059669 !important;
        border-radius: 12px !important;
        padding: 9px 13px !important;
        margin: 6px 0 10px 32px !important;
        font-size: 12.5px;
        color: #6EE7B7 !important;
    }

    /* 8. Bottom Chat Input - Completely eliminate dark black wrapper */
    div[data-testid="stChatInput"],
    .stChatInput,
    div[data-testid="stChatInputContainer"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-header) div[data-testid="stChatInput"] {
        padding: 10px 14px 14px 14px !important;
        background-color: #202836 !important;
        background: #202836 !important;
        border-top: 1px solid #2E3A4E !important;
        margin: 0 !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] > div,
    .stChatInput > div {
        border-radius: 22px !important;
        border: 1.5px solid #2E3A4E !important;
        background-color: #181F2B !important;
        background: #181F2B !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    }

    div[data-testid="stChatInput"] textarea,
    .stChatInput textarea {
        background-color: transparent !important;
        background: transparent !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 13.5px !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder,
    .stChatInput textarea::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
    }

    div[data-testid="stChatInput"] > div:focus-within,
    .stChatInput > div:focus-within {
        border-color: #9F8FEF !important;
        background-color: #181F2B !important;
        background: #181F2B !important;
        box-shadow: 0 0 0 2px rgba(159, 143, 239, 0.25) !important;
    }

    div[data-testid="stChatInput"] button,
    .stChatInput button {
        background-color: transparent !important;
        background: transparent !important;
        color: #9F8FEF !important;
        border: none !important;
    }

    div[data-testid="stChatInput"] button svg,
    .stChatInput button svg {
        fill: #9F8FEF !important;
    }

    /* Admin View Card */
    .admin-card-header {
        border-bottom: 1px solid #2E3A4E;
        padding-bottom: 14px;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. Session State Initialization
# ------------------------------------------------------------------------------
if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"  # "chat" or "admin"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! How can I help you with CarePal's home care services today?",
            "citations": [],
            "is_fallback": False,
            "show_starters": True,
        }
    ]

if "leads_submitted" not in st.session_state:
    st.session_state.leads_submitted = set()

# ------------------------------------------------------------------------------
# 4. Top Navigation Bar (Page Top-Right Admin Button)
# ------------------------------------------------------------------------------
nav_col_left, nav_col_right = st.columns([10, 2.2])

with nav_col_left:
    render_html("""
    <div class="top-nav-bar">
        <div class="brand-title">
            <span>TheCarePal</span>
            <span class="brand-subtitle">• AI Support Assistant</span>
        </div>
    </div>
    """)

with nav_col_right:
    if st.session_state.current_view == "chat":
        if st.button("📊 Ops Admin", key="btn_top_admin"):
            st.session_state.current_view = "admin"
            st.rerun()
    else:
        if st.button("💬 Live Chat", key="btn_top_chat"):
            st.session_state.current_view = "chat"
            st.rerun()

# ------------------------------------------------------------------------------
# 5. Helpers for Clean Citation Pills & Text
# ------------------------------------------------------------------------------
def extract_clean_citations(sources: list) -> list:
    """Extracts unique title & url pairs for citation pills."""
    seen_urls = set()
    pills = []
    for s in sources:
        url = s.get("page_url", "")
        title = s.get("page_title", "CarePal Info")
        short_title = title.split("|")[0].strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            pills.append({"title": short_title, "url": url})
    return pills


def clean_content_text(text: str) -> str:
    """Strips raw markdown citation lists and robotic disclaimers."""
    if "\n\n---\n**📚 Sources Cited:**" in text:
        text = text.split("\n\n---\n**📚 Sources Cited:**")[0]
    elif "\n\n---\n" in text:
        text = text.split("\n\n---\n")[0]

    # Remove robotic disclaimers in English, Hindi, and Hinglish
    disclaimer_patterns = [
        r"^according\s*to\s*thecarepal'?s?\s*published\s*website\s*information\s*:?\s*",
        r"^thecarepal\s*ki\s*published\s*website\s*information\s*ke\s*mutabik\s*(\([^\)]*\))?\s*:?\s*",
        r"^thecarepal\s*की\s*प्रकाशित\s*जानकारी\s*के\s*अनुसार\s*(\([^\)]*\))?\s*:?\s*",
        r"^as\s*per\s*thecarepal'?s?\s*website\s*:?\s*",
        r"^based\s*on\s*thecarepal'?s?\s*published\s*information\s*:?\s*",
    ]
    for pat in disclaimer_patterns:
        text = re.sub(pat, "", text.strip(), flags=re.IGNORECASE)

    # Remove any stray html brackets
    text = re.sub(r"</?div[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def format_bot_text(text: str) -> str:
    """Safely formats bot markdown into polished, executive-ready HTML."""
    cleaned = clean_content_text(text)
    if not cleaned:
        return ""

    lines = cleaned.splitlines()
    html_lines = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            html_lines.append('<div class="bot-spacer"></div>')
            continue

        # Check for headings (#, ##, ###)
        heading_match = re.match(r"^#{1,4}\s+(.+)$", line_str)
        if heading_match:
            h_text = html.escape(heading_match.group(1))
            h_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", h_text)
            html_lines.append(f'<div class="bot-heading">{h_text}</div>')
            continue

        # Check for bullet points (•, *, -)
        bullet_match = re.match(r"^[\•\*\-]\s+(.+)$", line_str)
        if bullet_match:
            b_text = html.escape(bullet_match.group(1))
            b_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", b_text)
            b_text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", b_text)
            html_lines.append(f'<div class="bot-bullet"><span class="bot-bullet-dot">●</span><span>{b_text}</span></div>')
            continue

        # Check for numbered lists (1. , 2. )
        num_match = re.match(r"^(\d+[\.\)])\s+(.+)$", line_str)
        if num_match:
            n_num = html.escape(num_match.group(1))
            n_text = html.escape(num_match.group(2))
            n_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", n_text)
            html_lines.append(f'<div class="bot-bullet"><span class="bot-bullet-dot" style="font-weight:600; font-size:12px;">{n_num}</span><span>{n_text}</span></div>')
            continue

        # Normal text line
        escaped_line = html.escape(line_str)
        escaped_line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped_line)
        escaped_line = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped_line)
        html_lines.append(f'<div class="bot-line">{escaped_line}</div>')

    return "".join(html_lines)


# ==============================================================================
# VIEW 1: CONVERSATIONAL CHAT WIDGET
# ==============================================================================
if st.session_state.current_view == "chat":
    # Center the widget card using 3 responsive columns
    col_l, col_center, col_r = st.columns([1, 1.4, 1])

    with col_center:
        # Outer Card Container (#202836)
        with st.container(border=True):
            # 1. Slate & Periwinkle Header Banner with Teal Status
            render_html("""
            <div class="chat-header">
                <div class="header-left">
                    <div class="bot-avatar-circle">🤖</div>
                    <div class="header-title-box">
                        <div class="header-title">CarePal</div>
                        <div class="header-status"><span class="status-dot"></span>Online Now</div>
                    </div>
                </div>
            </div>
            """)

            clicked_starter_query = None

            # 2. Scrollable Messages Area
            chat_container = st.container(height=460)
            with chat_container:
                for msg_idx, msg in enumerate(st.session_state.messages):
                    role = msg["role"]
                    raw_content = msg["content"]
                    citations = msg.get("citations", [])
                    is_fallback = msg.get("is_fallback", False)
                    show_starters = msg.get("show_starters", False)

                    if role == "assistant":
                        formatted_html = format_bot_text(raw_content)

                        # Citations pills HTML
                        citations_html = ""
                        if citations:
                            pills_html = []
                            for cit in citations:
                                pills_html.append(f'<a href="{cit["url"]}" target="_blank" class="citation-pill">📄 {html.escape(cit["title"])} ↗</a>')
                            citations_html = f'<div class="citations-row">{"".join(pills_html)}</div>'

                        # Compact, single-line bot bubble HTML with zero inter-tag whitespace
                        bot_bubble_html = (
                            '<div class="bot-msg-wrapper">'
                            '<div class="bot-icon-small">🤖</div>'
                            '<div class="bot-content-col">'
                            '<span class="bot-author-label">CarePal</span>'
                            f'<div class="bot-bubble">{formatted_html}</div>'
                            f'{citations_html}'
                            '</div></div>'
                        )
                        render_html(bot_bubble_html)

                        # Quick Starter Pills (below initial greeting only)
                        if show_starters and len(st.session_state.messages) == 1:
                            starters = [
                                ("Home Care Services", "What home care services does TheCarePal provide?"),
                                ("Refund Policy (हिंदी)", "क्या रिफंड मिलता है अगर हम सर्विस कैंसिल करें?"),
                                ("Baby & Elderly (Hinglish)", "Baby care aur elderly care mein kya facilities milti hain?"),
                                ("Pricing / ICU Setup", "How much does a 12-hour ICU ventilator setup cost at home?"),
                            ]
                            s1, s2 = st.columns(2)
                            if s1.button(starters[0][0], key="btn_s1", use_container_width=True):
                                clicked_starter_query = starters[0][1]
                            if s2.button(starters[1][0], key="btn_s2", use_container_width=True):
                                clicked_starter_query = starters[1][1]

                            s3, s4 = st.columns(2)
                            if s3.button(starters[2][0], key="btn_s3", use_container_width=True):
                                clicked_starter_query = starters[2][1]
                            if s4.button(starters[3][0], key="btn_s4", use_container_width=True):
                                clicked_starter_query = starters[3][1]

                        # Inline Lead Capture Form for Fallback Out-of-Scope Queries
                        if is_fallback:
                            form_key = f"lead_form_{msg_idx}"
                            if form_key in st.session_state.leads_submitted:
                                render_html("""
                                <div class="lead-success-banner">
                                    ✅ <b>Callback Requested!</b> Our care coordinator will call you shortly.
                                </div>
                                """)
                            else:
                                with st.form(key=f"lead_form_submit_{msg_idx}", clear_on_submit=False, border=True):
                                    render_html('<div class="inline-lead-title">📞 Connect with CarePal Team</div>')
                                    lead_name = st.text_input("Name", placeholder="Enter your full name", label_visibility="collapsed")
                                    lead_phone = st.text_input("Phone Number", placeholder="10-digit mobile number", label_visibility="collapsed")
                                    submit_lead = st.form_submit_button("Request Callback", use_container_width=True)

                                    if submit_lead:
                                        clean_phone = re.sub(r"\D", "", lead_phone)
                                        if not lead_name.strip():
                                            st.error("Please enter your name.")
                                        elif len(clean_phone) != 10:
                                            st.error("Please enter a valid 10-digit mobile number.")
                                        else:
                                            db.save_lead(
                                                name=lead_name.strip(),
                                                phone=lead_phone.strip(),
                                                service_needed="General Care Callback",
                                                query_context=msg.get("query_context", "Out of scope query"),
                                            )
                                            st.session_state.leads_submitted.add(form_key)
                                            st.rerun()

                    elif role == "user":
                        user_text = html.escape(clean_content_text(raw_content)).replace("\n", "<br>")
                        user_bubble_html = f'<div class="user-msg-wrapper"><div class="user-bubble">{user_text}</div></div>'
                        render_html(user_bubble_html)

            # 3. Question Asking Input (Seamlessly integrated inside card)
            chat_input_val = st.chat_input("Reply to CarePal...")
            user_prompt = chat_input_val or clicked_starter_query

            if user_prompt:
                # Add user message to state
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_prompt,
                    "citations": [],
                    "is_fallback": False,
                })

                # Stream answer into chat container
                with chat_container:
                    # Render immediate user bubble
                    usr_html = html.escape(user_prompt).replace("\n", "<br>")
                    render_html(f'<div class="user-msg-wrapper"><div class="user-bubble">{usr_html}</div></div>')

                    # Placeholder for streaming bot bubble
                    live_placeholder = st.empty()
                    init_live = (
                        '<div class="bot-msg-wrapper">'
                        '<div class="bot-icon-small">🤖</div>'
                        '<div class="bot-content-col">'
                        '<span class="bot-author-label">CarePal</span>'
                        '<div class="bot-bubble"><i>CarePal is typing...</i></div>'
                        '</div></div>'
                    )
                    live_placeholder.markdown(init_live, unsafe_allow_html=True)

                    # Run RAG retrieval & streaming generator
                    gen, meta = rag_engine.answer_query(user_prompt, stream=True)

                    accumulated_chunks = []
                    for chunk in gen:
                        accumulated_chunks.append(chunk)
                        streamed_so_far = "".join(accumulated_chunks)
                        live_html = format_bot_text(streamed_so_far)
                        cur_live = (
                            '<div class="bot-msg-wrapper">'
                            '<div class="bot-icon-small">🤖</div>'
                            '<div class="bot-content-col">'
                            '<span class="bot-author-label">CarePal</span>'
                            f'<div class="bot-bubble">{live_html}</div>'
                            '</div></div>'
                        )
                        live_placeholder.markdown(cur_live, unsafe_allow_html=True)

                    full_answer = clean_content_text("".join(accumulated_chunks))

                    # Safety Guard: If raw OUT_OF_SCOPE token is present, convert into user-friendly callback offer
                    if "OUT_OF_SCOPE" in full_answer or full_answer.strip() == "OUT_OF_SCOPE":
                        meta["is_fallback"] = True
                        full_answer = rag_engine.get_fallback_response(meta.get("language", "English"))
                        citations_list = []
                    else:
                        citations_list = extract_clean_citations(meta.get("sources", [])) if not meta.get("is_fallback") else []

                    # Store final assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "citations": citations_list,
                        "is_fallback": meta.get("is_fallback", False),
                        "query_context": user_prompt,
                    })
                    st.rerun()


# ==============================================================================
# VIEW 2: OPERATIONS ADMIN VIEW (Simple Admin View)
# ==============================================================================
else:
    with st.container(border=True):
        render_html("""
        <div class="admin-card-header">
            <h2 style="color: #9F8FEF; margin: 0; font-size: 24px;">📊 CarePal Operations Admin</h2>
            <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                Real-time dashboard for unanswered questions, content gaps, and patient callback leads.
            </p>
        </div>
        """)

        # 1. Metric Cards Row
        unanswered_raw = db.get_unanswered_questions(limit=100)
        leads_raw = db.get_leads(limit=100)
        unanswered_df = pd.DataFrame(unanswered_raw)
        leads_df = pd.DataFrame(leads_raw)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="Unanswered Content Gaps",
                value=len(unanswered_df),
                help="Queries where similarity score was below threshold or intent was out of scope.",
            )
        with m2:
            st.metric(
                label="Callback Leads Captured",
                value=len(leads_df),
                help="Prospective patients who submitted Name + Phone for follow-up.",
            )
        with m3:
            st.metric(
                label="Total System Events Logged",
                value=len(unanswered_df) + len(leads_df),
                help="Total logged events in carepal_ops.db",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Tabbed Content
        tab_leads, tab_unanswered, tab_info = st.tabs([
            "📞 Patient Callback Leads",
            "❓ Unanswered Questions (Content Gaps)",
            "ℹ️ System & Vector DB Status",
        ])

        with tab_leads:
            st.markdown("#### Patient Leads Awaiting Contact")
            if leads_df.empty:
                st.info("No callback leads recorded yet. Try asking an out-of-scope question in the chat and submitting the form!")
            else:
                lead_cols = [c for c in ["id", "created_at", "name", "phone", "service_needed", "status", "query_context"] if c in leads_df.columns]
                display_leads = leads_df[lead_cols].copy()
                st.dataframe(
                    display_leads,
                    use_container_width=True,
                    column_config={
                        "id": st.column_config.NumberColumn("ID", width="small"),
                        "created_at": st.column_config.DatetimeColumn("Submitted At"),
                        "name": st.column_config.TextColumn("Patient Name"),
                        "phone": st.column_config.TextColumn("Phone Number"),
                        "service_needed": st.column_config.TextColumn("Service Category"),
                        "status": st.column_config.TextColumn("Status"),
                        "query_context": st.column_config.TextColumn("Original Question"),
                    },
                )
                csv_leads = display_leads.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Leads to CSV",
                    data=csv_leads,
                    file_name=f"carepal_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

        with tab_unanswered:
            st.markdown("#### Content Gaps (Questions the website could not answer)")
            st.caption("Use this table to identify which service details, pricing, or FAQ topics should be added to TheCarePal website next.")

            if unanswered_df.empty:
                st.info("No unanswered questions logged yet.")
            else:
                unans_cols = [c for c in ["id", "created_at", "question", "language", "retrieval_score", "reason", "status"] if c in unanswered_df.columns]
                display_unanswered = unanswered_df[unans_cols].copy()
                st.dataframe(
                    display_unanswered,
                    use_container_width=True,
                    column_config={
                        "id": st.column_config.NumberColumn("ID", width="small"),
                        "created_at": st.column_config.DatetimeColumn("Timestamp"),
                        "question": st.column_config.TextColumn("Question Asked"),
                        "language": st.column_config.TextColumn("Language"),
                        "retrieval_score": st.column_config.NumberColumn("Top Similarity", format="%.3f"),
                        "reason": st.column_config.TextColumn("Fallback Reason"),
                        "status": st.column_config.TextColumn("Status"),
                    },
                )
                csv_unans = display_unanswered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Content Gaps to CSV",
                    data=csv_unans,
                    file_name=f"carepal_content_gaps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

        with tab_info:
            st.markdown("#### Knowledge Base & Vector Index Status")
            st.write({
                "FAISS Index Path": str(config.VECTOR_INDEX_FILE),
                "Chunks Indexed": f"{rag_engine.vector_store.index.ntotal if rag_engine.vector_store.index else 0} vectors",
                "Embedding Model": config.EMBEDDING_MODEL_NAME,
                "Similarity Guardrail Threshold": config.SIMILARITY_THRESHOLD,
                "SQLite Database": str(config.SQLITE_DB_PATH),
                "LLM Engine": f"{config.GEMINI_MODEL_NAME} ({rag_engine.llm_sdk if rag_engine.llm_client else 'Local Extractive'})",
            })
