"""
Verification & CLI Test Script for CarePal RAG Engine (Phase 2).
Tests:
1. English In-Scope Retrieval & Citations
2. Hindi (Devanagari) Query Handling
3. Hinglish Query Handling
4. Out-of-Scope Guardrail & Fallback Trigger
5. SQLite Database Logging (Unanswered Qs & Leads)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db
from src.rag_engine import rag_engine

# Set stdout encoding to UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_tests():
    print("=" * 80)
    print("CarePal Multilingual RAG Engine & Database Verification (Phase 2)")
    print("=" * 80)

    test_cases = [
        {
            "name": "Test 1: English In-Scope Query (Refund Policy)",
            "query": "What is the refund policy for CarePal services?",
            "expected_lang": "English",
            "expect_fallback": False,
            "expected_citation_kw": "refund",
        },
        {
            "name": "Test 2: Hindi Devanagari In-Scope Query (Elderly Care)",
            "query": "बुजुर्गों की देखभाल (Elderly Care) में क्या-क्या शामिल है?",
            "expected_lang": "Hindi",
            "expect_fallback": False,
            "expected_citation_kw": "elderly",
        },
        {
            "name": "Test 3: Hinglish In-Scope Query (Dementia Care)",
            "query": "Kya dementia care ke liye trained nurse ya caregiver milti hai?",
            "expected_lang": "Hinglish",
            "expect_fallback": False,
            "expected_citation_kw": "dementia",
        },
        {
            "name": "Test 4: Out-of-Scope Query (Pet Grooming & Dog Walking)",
            "query": "Do you provide dog walking and pet grooming services?",
            "expected_lang": "English",
            "expect_fallback": True,
            "expected_citation_kw": None,
        },
        {
            "name": "Test 5: Hinglish Out-of-Scope Fallback Query (Arthritis Diagnosis)",
            "query": "mujhe lagta h mujhe artheritis h",
            "expected_lang": "Hinglish",
            "expect_fallback": True,
            "expected_citation_kw": None,
        },
    ]

    all_passed = True

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        print(f"Query: \"{tc['query']}\"")

        detected_lang = rag_engine.detect_language(tc["query"])
        print(f"• Detected Language: {detected_lang} (Expected: {tc['expected_lang']})")

        answer, meta = rag_engine.answer_query(tc["query"], stream=False)
        is_fallback = meta["is_fallback"]
        top_score = meta["top_score"]

        print(f"• Fallback Triggered: {is_fallback} (Expected: {tc['expect_fallback']})")
        print(f"• Top Vector Similarity Score: {top_score:.4f}")

        # Verification checks
        lang_ok = detected_lang == tc["expected_lang"]
        fallback_ok = is_fallback == tc["expect_fallback"]
        citation_ok = True

        if not tc["expect_fallback"]:
            citation_str = "".join([c["page_url"] for c in meta["sources"]]).lower()
            citation_ok = tc["expected_citation_kw"] in citation_str
            print(f"• Expected Citation Keyword ('{tc['expected_citation_kw']}'): {'FOUND' if citation_ok else 'MISSING'}")

        if lang_ok and fallback_ok and citation_ok:
            print("  >>> [PASS] Test Passed!")
        else:
            print("  >>> [FAIL] Test Failed!")
            all_passed = False

        print("\nGenerated Assistant Output Preview:")
        print("-" * 50)
        print(answer[:350] + ("..." if len(answer) > 350 else ""))
        print("-" * 50)

    # --------------------------------------------------------------------------
    # Test 5: Lead Capture & SQLite Logging
    # --------------------------------------------------------------------------
    print("\n--- Test 5: SQLite Database & Lead Capture Verification ---")
    unanswered_before = len(db.get_unanswered_questions())
    leads_before = len(db.get_leads())

    # Simulate customer submitting callback form after out-of-scope query
    test_lead_id = db.save_lead(
        name="Sunita Rao",
        phone="+91 9812345678",
        service_needed="Dementia Care",
        query_context="Query about pet grooming (unsupported)",
    )
    print(f"• Saved Lead ID: {test_lead_id}")

    unanswered_list = db.get_unanswered_questions()
    leads_list = db.get_leads()
    metrics = db.get_metrics()

    print(f"• Unanswered Questions in DB: {len(unanswered_list)}")
    print(f"• Leads in DB: {len(leads_list)}")
    print(f"• Operations Metrics: {metrics}")

    if len(leads_list) > leads_before and len(unanswered_list) > 0:
        print("  >>> [PASS] SQLite Logging & Lead Capture Verified!")
    else:
        print("  >>> [FAIL] SQLite Logging Failed!")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print(">>> ALL PHASE 2 ENGINE TESTS PASSED (100% Verified)")
    else:
        print(">>> SOME TESTS FAILED")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
