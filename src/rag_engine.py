"""
Multilingual RAG Engine for CarePal Support Assistant (Phase 2).
Handles English, Hindi (Devanagari), and Hinglish queries, performs FAISS
dense retrieval with citation metadata, enforces strict factual guardrails,
triggers out-of-scope fallback with lead capture, and supports streaming responses.
"""

import sys
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db
from src.ingest import CarePalVectorStore

# Set stdout encoding to UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Common Hindi (Devanagari) and Hinglish terms mapped to English search concepts
MULTILINGUAL_KEYWORDS_MAP = {
    # Devanagari Hindi mappings
    "बुजुर्ग": "elderly care senior citizen home support mobility",
    "बुज़ुर्ग": "elderly care senior citizen home support mobility",
    "वृद्ध": "elderly care senior citizen home support mobility",
    "देखभाल": "care assistance support home services",
    "रीफ़ंड": "refund policy money back return cancellation payments",
    "रिफंड": "refund policy money back return cancellation payments",
    "वापसी": "refund return cancellation payments",
    "डिमेंशिया": "dementia memory loss care specialists patient safety",
    "अल्जाइमर": "alzheimers disease memory care support routine",
    "याददाश्त": "dementia alzheimers memory loss care",
    "भूलने": "dementia alzheimers memory care",
    "शिशु": "baby care newborn infant feeding mother support",
    "बच्चा": "baby care newborn infant feeding support",
    "बच्चे": "baby care newborn infant feeding support",
    "डॉक्टर": "doctor consultation online physician medical advice appointment",
    "परामर्श": "consultation doctor nurse therapy appointment",
    "दवा": "pharmaceutical medication delivery genuine medicine",
    "दवाई": "pharmaceutical medication delivery genuine medicine",
    "दवाइयां": "pharmaceutical medication delivery medicine refill",
    "आपातकालीन": "emergency care rapid response ambulance urgent",
    "इमरजेंसी": "emergency care rapid response ambulance urgent",
    "थेरेपी": "mental therapy counseling depression anxiety",
    "काउंसलिंग": "mental therapy counseling depression anxiety",
    "लाइव-इन": "24/7 live-in care full-time caregiver home support",
    "खर्च": "pricing service costs charges consultation fee",
    "शुल्क": "pricing service costs charges consultation fee",
    "फीस": "pricing service costs charges consultation fee",
    "संपर्क": "contact us phone email care coordinator support team",
    "फोन": "contact us phone number customer support",
    "नंबर": "contact us phone number call",

    # Hinglish & Romanized Hindi mappings
    "refund": "refund policy money back return cancellation payment",
    "paisa": "refund payment charges cost",
    "paise": "refund payment charges cost",
    "wapas": "refund return cancellation",
    "vapas": "refund return cancellation",
    "buzurg": "elderly care senior citizen old age support",
    "bujurg": "elderly care senior citizen old age support",
    "elderly": "elderly care senior citizen home support",
    "parents": "elderly care senior citizen support at home",
    "maa": "elderly care senior citizen support",
    "baap": "elderly care senior citizen support",
    "dementia": "dementia memory loss care specialists patient safety",
    "alzheimer": "alzheimers disease memory care support routine",
    "yaad": "dementia alzheimers memory loss care",
    "yaaddasht": "dementia alzheimers memory loss care",
    "bhool": "dementia alzheimers memory care",
    "live-in": "24/7 live-in care full-time caregiver home support shifts",
    "livein": "24/7 live-in care full-time caregiver home support shifts",
    "24": "24/7 live-in care round the clock support",
    "nurse": "nurse consultation home nursing caregiver medical staff",
    "doctor": "doctor consultation online physician medical advice appointment",
    "baby": "baby care newborn infant feeding mother support",
    "baccha": "baby care newborn infant feeding support",
    "bache": "baby care newborn infant feeding support",
    "autism": "autism spectrum behavioral therapy child care",
    "therapy": "mental therapy counseling depression anxiety psychologist",
    "emergency": "emergency care rapid response ambulance urgent medical",
    "dawai": "pharmaceutical medication delivery genuine medicine",
    "dawiyan": "pharmaceutical medication delivery medicine refill",
    "medicine": "pharmaceutical medication prescription delivery",
    "price": "pricing service costs charges consultation fee",
    "kharcha": "pricing costs fees charges",
    "kitna": "pricing charges costs fee",
    "contact": "contact us phone email care coordinator support team",
    "phone": "contact us phone number customer support",
    "number": "contact us phone number call",
    "call": "contact us phone arrangement consultation booking",
    "lisa": "lisa voice assistant mobile app features",
    "app": "the carepal mobile app booking features",
}

# Known out-of-scope intent triggers (services CarePal definitely does not offer)
OUT_OF_SCOPE_PATTERNS = [
    r"\b(dog|cat|pet|puppy)\b",
    r"\b(car|bike|vehicle|mechanic|repair)\b",
    r"\b(crypto|bitcoin|stock|trading|shares)\b",
    r"\b(hotel|flight|train|travel booking)\b",
    r"\b(plumber|carpenter|electrician)\b",
    r"\b(haircut|salon|spa|beauty)\b",
    r"\b(legal case|lawyer|court)\b",
    r"\b(real\s*estate|property|flat|plot|broker|zameen|makaan|ghar\s*khareed)\b",
    r"\b(icu|ventilator)\b",
    r"\b(artheritis|arthritis)\b",
    # Devanagari Hindi out-of-scope triggers
    r"(कुत्ता|बिल्ली|पालतू|गाड़ी|कार|बाइक|मैकेनिक|क्रिप्टो|बिटकॉइन|शेयर|होटल|फ्लाइट|प्लंबर|कारपेंटर|सैलून|पार्लर|वकील|अदालत|जमीन|प्लॉट|फ्लैट|मकान|वेंटीलेटर|गठिया)",
]


class RAGEngine:
    """Multilingual RAG Engine with guardrails and streaming LLM generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = config.GEMINI_MODEL_NAME,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
        top_k: int = config.DEFAULT_TOP_K,
    ):
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        # Load vector store
        self.vector_store = CarePalVectorStore()
        self.vector_store_loaded = self.vector_store.load()

        # Initialize LLM client
        self.llm_client = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initializes Google GenAI / Gemini client if API key is provided."""
        if not self.api_key:
            return

        try:
            # Try official google-genai SDK first
            from google import genai
            self.llm_client = genai.Client(api_key=self.api_key)
            self.llm_sdk = "google-genai"
        except Exception:
            try:
                # Fallback to google.generativeai SDK
                import google.generativeai as ggenai
                ggenai.configure(api_key=self.api_key)
                self.llm_client = ggenai.GenerativeModel(self.model_name)
                self.llm_sdk = "google-generativeai"
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini SDK: {e}")
                self.llm_client = None

    def set_api_key(self, api_key: str) -> None:
        """Update API key dynamically (e.g. from Streamlit sidebar)."""
        self.api_key = api_key.strip()
        self._init_llm_client()

    # --------------------------------------------------------------------------
    # Language Detection & Query Translation
    # --------------------------------------------------------------------------
    @staticmethod
    def detect_language(text: str) -> str:
        """Detects if text is in Hindi (Devanagari), Hinglish, or English."""
        # Devanagari Unicode block: U+0900 to U+097F
        if re.search(r"[\u0900-\u097F]", text):
            return "Hindi"

        # Distinctive Hindi grammar markers & colloquial vocabulary in Roman script
        # Note: Pure English terms like 'refund', 'doctor', 'baby' are deliberately omitted
        # so standard English queries are never misclassified as Hinglish.
        hinglish_markers = {
            # Pronouns & pointers
            "mujhe", "mujhko", "mera", "meri", "mere", "main", "hum", "humein",
            "aap", "aapka", "aapki", "aapke", "tum", "tumhara", "tumhe", "yeh", "ye",
            "woh", "wo", "kisi", "kisko", "apna", "apne", "apni", "unhe", "unka",
            # Question words
            "kya", "kyu", "kyun", "kaise", "kahan", "kidhar", "kab", "kitna",
            "kitni", "kitne", "kaun", "kon",
            # Auxiliary verbs & common stems
            "hai", "hain", "h", "ha", "ho", "tha", "thi", "the", "hoga", "hogi", "honge",
            "karna", "karo", "karein", "karen", "kare", "karta", "karti", "karte",
            "lagta", "lagti", "lagte", "chahiye", "chahie", "chahta", "chahti",
            "milega", "milegi", "milte", "milti", "milta", "batao", "bataiye", "bataie",
            "denge", "dega", "degi", "lena", "lo", "raha", "rahi", "rahe",
            # Particles, prepositions, conjunctions
            "ke", "ki", "ka", "ko", "se", "par", "pe", "me", "mein", "tak", "bhi",
            "aur", "ya", "lekin", "magar", "liye", "waala", "wali", "wale", "nahi", "nahin", "na",
            # Distinctive Hindi nouns (not standard English words)
            "buzurg", "bujurg", "baccha", "bacha", "bache", "dawai", "dawiyan", "bimar",
            "bimari", "ilaj", "ilaaj", "dard", "takleef", "dikkat", "madad", "paisa", "paise", "wapas",
        }
        tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
        matched = tokens.intersection(hinglish_markers)
        if len(matched) >= 2 or (len(tokens) <= 6 and len(matched) >= 1):
            return "Hinglish"

        return "English"

    def normalize_query_for_retrieval(self, query: str, language: str) -> str:
        """
        Translates or expands non-English queries into English keywords to
        maximize semantic similarity against English document chunks.
        """
        query_lower = query.lower()

        # If LLM is available, we can use it to get a clean English translation query
        # But a fast deterministic keyword translation is instantaneous and $0 cost:
        expanded_terms = []
        for term, expansion in MULTILINGUAL_KEYWORDS_MAP.items():
            if term in query_lower:
                expanded_terms.append(expansion)

        if expanded_terms:
            # Append English concepts to the search query
            combined = f"{query} {' '.join(expanded_terms[:3])}"
            return combined

        return query

    # --------------------------------------------------------------------------
    # Retrieval & Out-of-Scope Detection
    # --------------------------------------------------------------------------
    def is_definitely_out_of_scope(self, query: str) -> bool:
        """Quick regex check for completely unrelated service requests."""
        query_lower = query.lower()
        for pat in OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, query_lower):
                return True
        return False

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict], float]:
        """
        Executes semantic vector search and returns ranked chunks with top score.
        Uses hybrid multi-topic retrieval:
        1. Primary vector search on original query.
        2. Keyword-normalized vector search.
        3. Intent & topic-targeted vector search for multi-topic queries (e.g. asking about
           caregiver booking for elderly/baby care AND refund/cancellation policies).
        All candidate chunks are merged and deduplicated by chunk_id.
        """
        if not self.vector_store_loaded:
            self.vector_store_loaded = self.vector_store.load()

        k = top_k or self.top_k
        lang = self.detect_language(query)
        search_query = self.normalize_query_for_retrieval(query, lang)
        ql = query.lower()

        all_candidates = []

        # 1. Primary vector search
        results_primary = self.vector_store.search(query, top_k=k)
        all_candidates.extend(results_primary)

        # 2. Keyword-normalized search
        if search_query != query:
            results_expanded = self.vector_store.search(search_query, top_k=k)
            all_candidates.extend(results_expanded)

        # 3. Topic-targeted retrieval for multi-intent / multi-topic queries
        topic_queries = {
            "refund": ("refund policy cancellation return fee charges money back", ["refund", "cancel", "paisa", "paise", "wapas", "vapas", "return", "रिफंड", "वापसी", "रद्दीकरण"]),
            "elderly": ("elderly care senior citizen old age mobility support", ["elderly", "senior", "buzurg", "bujurg", "old age", "parents", "बुजुर्ग", "वृद्ध"]),
            "babycare": ("baby care newborn infant feeding support mother", ["baby", "infant", "newborn", "baccha", "bacha", "bache", "shishu", "शिशु", "बच्चा", "बच्चे"]),
            "dementia": ("dementia memory loss alzheimers care routine", ["dementia", "alzheimer", "memory", "yaad", "bhool", "डिमेंशिया", "अल्जाइमर", "याददाश्त"]),
            "livein": ("24/7 live-in care full time caregiver attendant", ["live-in", "livein", "24/7", "full-time", "attendant", "लाइव-इन", "अटेंडेंट"]),
            "doctor": ("online doctor consultation physician medical advice appointment", ["doctor", "physician", "consultation", "paramarsh", "डॉक्टर", "परामर्श"]),
            "careers": ("careers join carepal caregiver nurse job openings", ["career", "job", "hiring", "vacancy", "join", "नौकरी"]),
        }

        for topic, (search_str, keywords) in topic_queries.items():
            if any(kw in ql for kw in keywords):
                topic_res = self.vector_store.search(search_str, top_k=2)
                all_candidates.extend(topic_res)

        # Merge results prioritizing highest scores while maintaining chunk uniqueness
        seen_chunk_ids = set()
        merged_results = []
        all_candidates_sorted = sorted(all_candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        for cand in all_candidates_sorted:
            cid = cand.get("chunk_id")
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                merged_results.append(cand)
                if len(merged_results) >= k:
                    break

        top_score = merged_results[0]["score"] if merged_results else 0.0
        return merged_results, top_score

    # --------------------------------------------------------------------------
    # Conversational Intent Handler
    # --------------------------------------------------------------------------
    def handle_conversational_intent(self, query: str, detected_language: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Handles greetings, language switching, bot identity, and common chit-chat
        gracefully so conversational queries do not trigger false-positive out-of-scope
        fallbacks or prompt unnecessary lead collection.
        Returns:
            (response_text, response_language) or (None, None)
        """
        ql = query.lower().strip()
        q_clean = re.sub(r"[\?\.\!\,]+$", "", ql).strip()

        # 1. English language inquiries / requests
        if re.search(r"\b(can\s*(you|we)|do\s*you|please)?\s*(talk|speak|chat|write|reply|respond|converse|baat)\s*(in\s*|to\s*me\s*in\s*)?english\b", q_clean) or re.search(r"^(can\s*we\s*talk\s*in\s*english|speak\s*english|english\s*(please|me|mein)?)$", q_clean):
            resp = (
                "Yes, absolutely! I can converse fluently in English, Hindi, and Hinglish. "
                "How can I assist you with CarePal's home care services today?"
            )
            return resp, "English"

        # 2. Hindi language inquiries / requests
        if "हिंदी" in q_clean or re.search(r"\b(hindi\s*(mai|me|mein)?\s*(baat|bol|bolte|bolna)|kya\s*aap\s*hindi|can\s*(you|we)\s*(speak|talk|chat)\s*(in\s*)?hindi)\b", q_clean) or re.search(r"^(hindi\s*(please|me|mein)?)$", q_clean):
            resp = (
                "हाँ, बिल्कुल! मैं हिंदी (Hindi), हिंग्लिश (Hinglish) और अंग्रेज़ी (English) में बात कर सकता हूँ। "
                "आप CarePal की होम केयर सेवाओं (जैसे बुजुर्गों की देखभाल, डिमेंशिया केयर, नवजात शिशु देखभाल, या ऑनलाइन डॉक्टर परामर्श) के बारे में क्या जानना चाहते हैं?"
            )
            return resp, "Hindi"

        # 3. Hinglish language inquiries / requests
        if re.search(r"\b(hinglish\s*(mai|me|mein)?\s*baat|can\s*(you|we)\s*(speak|talk|chat)\s*(in\s*)?hinglish)\b", q_clean) or re.search(r"^(hinglish\s*(please|me|mein)?)$", q_clean):
            resp = (
                "Haan bilkul! Main Hinglish, Hindi aur English teeno me baat kar sakta hoon. "
                "Aap CarePal ki home care services (elderly care, dementia care, baby care, ya doctor consultation) ke baare me kya janna chahte hain?"
            )
            return resp, "Hinglish"

        # 4. Pure Greetings & Well-being
        if (
            re.search(r"^(hi|hello|hey|namaste|namaskar|pranam|good\s*(morning|afternoon|evening))\b", q_clean)
            or re.search(r"\b(how\s*are\s*you|how's\s*it\s*going|kaise\s*ho|kya\s*(hal|haal|chal)\s*(hai|h)?|aap\s*kaise\s*(ho|hain|h)|sab\s*thik)\b", q_clean)
        ) and len(q_clean.split()) <= 6:
            if detected_language == "Hindi":
                resp = (
                    "नमस्ते! मैं बहुत अच्छा हूँ, पूछने के लिए धन्यवाद। CarePal होम केयर असिस्टेंट में आपका स्वागत है। "
                    "मैं आपकी या आपके परिवार की घरेलू स्वास्थ्य देखभाल सेवाओं (बुजुर्गों की देखभाल, नवजात शिशु केयर, डिमेंशिया सहायता) में कैसे मदद कर सकता हूँ?"
                )
            elif detected_language == "Hinglish":
                resp = (
                    "Hello! Main bilkul theek hoon, thank you! CarePal support assistant me aapka swagat hai. "
                    "Main aapke parivaar ke liye home care services (elderly care, baby care, nursing support) me kaise help kar sakta hoon?"
                )
            else:
                resp = (
                    "Hello! I am doing great, thank you for asking. Welcome to CarePal! "
                    "How can I help you or your loved ones with our in-home care services today?"
                )
            return resp, detected_language

        # 5. Identity & Bot capabilities
        if re.search(r"\b(who\s*are\s*you|what\s*(are\s*you|can\s*you\s*do)|what\s*is\s*carepal|aap\s*(kaun|kon)\s*(ho|hain|h)|tum\s*(kaun|kon)\s*ho)\b", q_clean):
            if detected_language == "Hindi":
                resp = (
                    "मैं CarePal का AI सपोर्ट असिस्टेंट हूँ। मैं आपको TheCarePal की सभी होम केयर सेवाओं "
                    "(बुजुर्गों की देखभाल, अल्जाइमर और डिमेंशिया सपोर्ट, नवजात शिशु देखभाल, 24/7 लाइव-इन केयरगिवर्स, और ऑनलाइन डॉक्टर परामर्श) "
                    "के बारे में सटीक जानकारी देने के लिए यहाँ हूँ। आप मुझसे कोई भी सवाल पूछ सकते हैं!"
                )
            elif detected_language == "Hinglish":
                resp = (
                    "Main CarePal ka AI Support Assistant hoon! Main aapko TheCarePal ki certified home care services "
                    "(jaise elderly care, Alzheimer's & dementia support, newborn baby care, 24/7 live-in caregivers, aur online doctor consultation) "
                    "ke baare me poori details provide kar sakta hoon. Aap mujhse English, Hindi ya Hinglish me kuch bhi pooch sakte hain!"
                )
            else:
                resp = (
                    "I am CarePal's AI Support Assistant for TheCarePal (www.thecarepal.com). "
                    "I provide verified information about our in-home care services across India—including trained caregivers for seniors and dementia patients, newborn baby care, 24/7 live-in attendants, and online doctor consultations. "
                    "How can I help you today?"
                )
            return resp, detected_language

        # 6. Gratitude & Farewell
        if re.search(r"^(thank\s*you|thanks|thanku|dhanyawad|dhanyavad|shukriya|shukriyaa|bye|goodbye)\b", q_clean) and len(q_clean.split()) <= 5:
            if detected_language == "Hindi":
                resp = (
                    "आपका बहुत-बहुत धन्यवाद! यदि आपको या आपके परिवार को कभी भी विश्वसनीय होम केयर सेवाओं की आवश्यकता हो, "
                    "तो CarePal हमेशा आपकी सेवा में उपस्थित है। आपका दिन शुभ हो!"
                )
            elif detected_language == "Hinglish":
                resp = (
                    "Aapka bahut-bahut shukriya! Agar aapko kabhi bhi trustworthy home care services ki zaroorat ho, "
                    "toh CarePal hamesha aapki help ke liye ready hai. Have a wonderful day!"
                )
            else:
                resp = (
                    "You're very welcome! If you or your family ever need compassionate, verified home care support, "
                    "CarePal is always here to help. Have a wonderful day!"
                )
            return resp, detected_language

        return None, None

    # --------------------------------------------------------------------------
    # Fallback Generation
    # --------------------------------------------------------------------------
    def get_fallback_response(self, language: str) -> str:
        """Standardized polite refusal with callback offer across 3 languages."""
        if language == "Hindi":
            return (
                "मुझे हमारी वेबसाइट पर इस बारे में जानकारी नहीं मिली। "
                "लेकिन हमारी केयर कोऑर्डिनेशन टीम आपकी पूरी सहायता करने के लिए तैयार है!\n\n"
                "📞 **क्या आप चाहते हैं कि हमारी टीम आपसे संपर्क करे?**\n"
                "कृपया नीचे अपना **नाम** और **फ़ोन नंबर** साझा करें, "
                "और हमारे केयर मैनेजर जल्द ही आपको कॉल करेंगे।"
            )
        elif language == "Hinglish":
            return (
                "Mujhe hamari website par is baare me koi jankari nahi mili. "
                "Lekin hamari care team aapki puri help karne ke liye taiyar hai!\n\n"
                "📞 **Kya aap chahenge ki hamari team aapse contact kare?**\n"
                "Kripya niche apna **Name** aur **Phone Number** share karein, "
                "aur hamare care manager aapse jald hi connect karenge."
            )
        else:
            return (
                "I don't have this information in our published website content. "
                "However, our CarePal care coordination team would be glad to help you directly!\n\n"
                "📞 **Would you like our team to give you a call?**\n"
                "Please share your **Name** and **Phone Number** below, "
                "and a care manager will reach out shortly."
            )

    # --------------------------------------------------------------------------
    # Prompt Construction
    # --------------------------------------------------------------------------
    def _build_system_prompt(self, language: str) -> str:
        """Strict grounding system instructions."""
        lang_instruction = {
            "Hindi": "Respond in natural, polite Hindi (Devanagari script).",
            "Hinglish": "Respond in natural, polite conversational Hinglish (Roman script Hindi mixed with English), popular in urban India.",
            "English": "Respond in clear, compassionate professional English.",
        }.get(language, "Respond in clear, compassionate English.")

        return f"""You are the official CarePal Support Assistant for TheCarePal (www.thecarepal.com), an Indian home-care platform.
TheCarePal provides home care across India: trained caregivers and nurses for elderly care, dementia and Alzheimer's care, baby care, 24/7 live-in care, and online doctor consultations.

CRITICAL INSTRUCTIONS:
1. Answer the user's question directly using the verified CarePal context chunks below.
2. If the user asks about multiple topics (e.g. services, booking, and cancellation/refunds), synthesize the details from all relevant chunks provided.
3. If an inquiry is completely unrelated to CarePal or home healthcare (e.g. car mechanics, dog walking, crypto investments, divorce lawyers), reply with: "OUT_OF_SCOPE".
4. DO NOT hallucinate, assume, or invent services, pricing, or policies not stated in the context.
5. {lang_instruction}
6. Keep your tone empathetic, helpful, and concise. Format points with clear bullet points (•) where helpful.
7. Do NOT mention chunk IDs or raw system details.
8. NEVER prefix your response with robotic disclaimers such as "According to TheCarePal's published website information:", "Based on the website", or "As per our records". Answer directly, warmly, and naturally as CarePal's care coordinator.
9. Use clean Markdown: bold terms (**Heading**) for clarity and bullet points (•) for lists.
"""

    def _build_context_text(self, chunks: List[Dict]) -> str:
        """Formats retrieved chunks into numbered context blocks."""
        blocks = []
        for i, c in enumerate(chunks, start=1):
            title = c.get("page_title", "The CarePal")
            url = c.get("page_url", "")
            content = c.get("content", "").strip()
            blocks.append(f"[Source {i}: {title} | URL: {url}]\n{content}")
        return "\n\n---\n\n".join(blocks)

    def _format_citations(self, chunks: List[Dict]) -> str:
        """Extracts unique source citations in markdown format."""
        seen = set()
        citations = []
        for c in chunks:
            url = c.get("page_url", "")
            title = c.get("page_title", "The CarePal")
            if url and url not in seen:
                seen.add(url)
                citations.append(f"- [{title}]({url})")

        if not citations:
            return ""

        return "\n\n---\n**📚 Sources Cited:**\n" + "\n".join(citations)

    # --------------------------------------------------------------------------
    # Local Rule-Based / Extractive Answering (When no API key is set)
    # --------------------------------------------------------------------------
    def _generate_local_answer(
        self,
        query: str,
        chunks: List[Dict],
        language: str,
    ) -> str:
        """
        High-precision local extractive answering engine.
        Synthesizes the most relevant excerpts across retrieved chunks and
        formats the response cleanly without robotic disclaimers or raw markdown hashes.
        """
        if not chunks:
            return self.get_fallback_response(language)

        collected_sections = []
        seen_sentences = set()
        seen_titles = set()

        for chunk in chunks:
            content = chunk.get("content", "").strip()
            title = chunk.get("page_title", "").split("|")[0].strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            section_points = []
            for line in content.splitlines():
                line_str = line.strip()
                if not line_str or line_str.startswith("---") or line_str.lower().startswith("sources:"):
                    continue
                # Clean leading markdown headers
                clean_line = re.sub(r"^#+\s*", "", line_str).strip()
                if not clean_line or len(clean_line) < 15:
                    continue
                
                # Deduplicate similar sentences
                sentence_key = clean_line[:40].lower()
                if sentence_key in seen_sentences:
                    continue
                seen_sentences.add(sentence_key)

                if clean_line.startswith("• ") or clean_line.startswith("- "):
                    section_points.append(f"• {clean_line.lstrip('•- ')}")
                elif len(clean_line) > 35:
                    section_points.append(f"• {clean_line}")

                if len(section_points) >= 3:
                    break

            if section_points:
                collected_sections.append(f"**{title}**\n" + "\n".join(section_points))
                if len(collected_sections) >= 4:
                    break

        if collected_sections:
            body_text = "\n\n".join(collected_sections)
        else:
            first_content = re.sub(r"^#+\s*", "", chunks[0].get("content", "")).strip()
            body_text = first_content[:400]

        answer = f"{body_text}{self._format_citations(chunks)}"
        return answer

    # --------------------------------------------------------------------------
    # Main RAG Answering (Streaming & Batch)
    # --------------------------------------------------------------------------
    def answer_query(
        self,
        query: str,
        stream: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes end-to-end RAG pipeline.
        Returns:
            (answer_text_or_generator, metadata_dict)
            where metadata_dict = {
                "is_fallback": bool,
                "language": str,
                "top_score": float,
                "sources": List[Dict],
                "unanswered_id": Optional[int]
            }
        """
        query_clean = query.strip()
        language = self.detect_language(query_clean)

        # 0. Conversational & Language Switching Intent Handler (Greetings, Language capability, Identity)
        ch_resp, ch_lang = self.handle_conversational_intent(query_clean, language)
        if ch_resp:
            db.log_chat(query_clean, ch_resp, ch_lang or language, sources_cited="", is_fallback=False)
            meta = {
                "is_fallback": False,
                "language": ch_lang or language,
                "top_score": 1.0,
                "sources": [],
                "unanswered_id": None,
            }
            if stream:
                def _stream_conv():
                    for word in ch_resp.split(" "):
                        yield word + " "
                        time.sleep(0.012)
                return _stream_conv(), meta
            return ch_resp, meta

        # 1. Check obvious out of scope
        if self.is_definitely_out_of_scope(query_clean):
            unanswered_id = db.log_unanswered_question(
                question=query_clean,
                language=language,
                score=0.0,
                reason="out_of_scope_intent",
            )
            fallback_text = self.get_fallback_response(language)
            meta = {
                "is_fallback": True,
                "language": language,
                "top_score": 0.0,
                "sources": [],
                "unanswered_id": unanswered_id,
            }
            if stream:
                def _stream_fallback():
                    for word in fallback_text.split(" "):
                        yield word + " "
                        time.sleep(0.015)
                return _stream_fallback(), meta
            return fallback_text, meta

        # 2. Vector Retrieval
        chunks, top_score = self.retrieve(query_clean, top_k=self.top_k)

        # 3. Guardrail: Low similarity threshold check
        if not chunks or top_score < self.similarity_threshold:
            unanswered_id = db.log_unanswered_question(
                question=query_clean,
                language=language,
                score=top_score,
                reason="low_similarity_score",
            )
            fallback_text = self.get_fallback_response(language)
            meta = {
                "is_fallback": True,
                "language": language,
                "top_score": top_score,
                "sources": chunks,
                "unanswered_id": unanswered_id,
            }
            if stream:
                def _stream_fallback():
                    for word in fallback_text.split(" "):
                        yield word + " "
                        time.sleep(0.015)
                return _stream_fallback(), meta
            return fallback_text, meta

        # 4. In-Scope: Generate Answer via Gemini or Local Extractive
        citations = self._format_citations(chunks)

        if self.llm_client:
            # Generate with Google Gemini
            system_prompt = self._build_system_prompt(language)
            context_text = self._build_context_text(chunks)
            user_prompt = f"User Question: {query_clean}\n\nContext Information:\n{context_text}"

            try:
                if stream:
                    # Streaming generation generator
                    def _stream_gemini():
                        full_acc = []
                        gemini_success = True
                        is_out_of_scope_flag = False
                        try:
                            if self.llm_sdk == "google-genai":
                                from google.genai import types
                                gen_config = types.GenerateContentConfig(
                                    temperature=0.2,
                                )
                                response_stream = self.llm_client.models.generate_content_stream(
                                    model=self.model_name,
                                    contents=f"{system_prompt}\n\n{user_prompt}",
                                    config=gen_config,
                                )
                                for chunk in response_stream:
                                    if chunk.text:
                                        full_acc.append(chunk.text)
                                        if "OUT_OF_SCOPE" in "".join(full_acc):
                                            is_out_of_scope_flag = True
                                            break
                                        yield chunk.text
                            else:
                                # google.generativeai
                                response_stream = self.llm_client.generate_content(
                                    f"{system_prompt}\n\n{user_prompt}",
                                    stream=True,
                                )
                                for chunk in response_stream:
                                    if chunk.text:
                                        full_acc.append(chunk.text)
                                        if "OUT_OF_SCOPE" in "".join(full_acc):
                                            is_out_of_scope_flag = True
                                            break
                                        yield chunk.text
                        except Exception as e:
                            print(f"[Warning] Gemini stream error: {e}. Falling back to local synthesis.")
                            gemini_success = False

                        if is_out_of_scope_flag:
                            meta["is_fallback"] = True
                            unanswered_id = db.log_unanswered_question(query_clean, language, top_score, "llm_out_of_scope")
                            meta["unanswered_id"] = unanswered_id
                            meta["sources"] = []
                            fallback_text = self.get_fallback_response(language)
                            for word in fallback_text.split(" "):
                                yield word + " "
                                time.sleep(0.015)
                            return

                        if not gemini_success:
                            local_ans = self._generate_local_answer(query_clean, chunks, language)
                            for word in local_ans.split(" "):
                                yield word + " "
                                time.sleep(0.01)
                            db.log_chat(query_clean, local_ans + citations, language, citations, is_fallback=False)
                            return

                        # Check if model returned OUT_OF_SCOPE
                        accumulated = "".join(full_acc).strip()
                        if "OUT_OF_SCOPE" in accumulated:
                            meta["is_fallback"] = True
                            db.log_unanswered_question(query_clean, language, top_score, "llm_out_of_scope")
                        else:
                            # Yield citations at the end of the stream
                            yield citations
                            db.log_chat(query_clean, accumulated + citations, language, citations, is_fallback=False)

                    meta = {
                        "is_fallback": False,
                        "language": language,
                        "top_score": top_score,
                        "sources": chunks,
                        "unanswered_id": None,
                    }
                    return _stream_gemini(), meta
                else:
                    # Non-streaming Gemini
                    if self.llm_sdk == "google-genai":
                        from google.genai import types
                        gen_config = types.GenerateContentConfig(
                            temperature=0.2,
                        )
                        resp = self.llm_client.models.generate_content(
                            model=self.model_name,
                            contents=f"{system_prompt}\n\n{user_prompt}",
                            config=gen_config,
                        )
                        raw_text = resp.text.strip() if resp.text else ""
                    else:
                        resp = self.llm_client.generate_content(f"{system_prompt}\n\n{user_prompt}")
                        raw_text = resp.text.strip() if resp.text else ""

                    if "OUT_OF_SCOPE" in raw_text or len(raw_text) < 15:
                        unanswered_id = db.log_unanswered_question(query_clean, language, top_score, "llm_out_of_scope")
                        fallback_text = self.get_fallback_response(language)
                        meta = {
                            "is_fallback": True,
                            "language": language,
                            "top_score": top_score,
                            "sources": chunks,
                            "unanswered_id": unanswered_id,
                        }
                        return fallback_text, meta

                    final_answer = raw_text + citations
                    db.log_chat(query_clean, final_answer, language, citations, is_fallback=False)
                    meta = {
                        "is_fallback": False,
                        "language": language,
                        "top_score": top_score,
                        "sources": chunks,
                        "unanswered_id": None,
                    }
                    return final_answer, meta
            except Exception as e:
                print(f"[Warning] Gemini API call failed: {e}. Falling back to local synthesizer.")
                # Fall through to local synthesis on API error

        # 5. Local Synthesizer (Zero API cost / offline mode)
        local_ans = self._generate_local_answer(query_clean, chunks, language)
        db.log_chat(query_clean, local_ans, language, citations, is_fallback=False)
        meta = {
            "is_fallback": False,
            "language": language,
            "top_score": top_score,
            "sources": chunks,
            "unanswered_id": None,
        }

        if stream:
            def _stream_local():
                for word in local_ans.split(" "):
                    yield word + " "
                    time.sleep(0.015)
            return _stream_local(), meta

        return local_ans, meta


# Global default engine instance
rag_engine = RAGEngine()
