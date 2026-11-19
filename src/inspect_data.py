"""
Data Inspector & Quality Validator for TheCarePal (Phase 1).
Inspects scraped pages from data/raw_pages.json, computes comprehensive content
statistics, and executes validation checks to confirm no empty React shells,
repeated navbar/footer boilerplate, or duplicate pages remain.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Set stdout encoding to UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Prohibited boilerplate signatures that should never leak into clean page text
NAVBAR_BOILERPLATE_PATTERNS = [
    "LinkedIn community",
    "Instagram account",
    "Youtube account",
    "Twitter page",
    "floating-chat-btn",
    "Educational resources and health learning materials for better care",
    "Doctor Consultation Expert medical advice",
    "Nurse Consultation Professional nursing advice",
    "Mental Therapy Mental health support",
    "Live-in Care 24/7 home care services",
    "Your browser does not support the video tag",
    "Quick Links Sign Up Support Blogs Careers",
    "Smarter health starts with CarePal",
]

# Required core URLs for CarePal business operations
REQUIRED_CORE_PAGES = [
    "/elderly",
    "/dementia",
    "/alzeihmer",
    "/livein-care",
    "/babycare",
    "/doctor",
    "/refund",
    "/terms",
    "/privacy",
    "/contact",
]


class DataInspector:
    """Performs statistical analysis and validation checks on scraped data."""

    def __init__(self, data_file: Path = config.RAW_PAGES_FILE):
        self.data_file = data_file
        self.pages: List[Dict] = []
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def load_data(self) -> bool:
        """Loads JSON data from file."""
        if not self.data_file.exists():
            print(f"[FATAL] Data file not found at: {self.data_file}")
            print("Please run `python src/scraper.py` first to generate raw_pages.json.")
            return False

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                self.pages = json.load(f)
            if not self.pages:
                print(f"[FATAL] Data file is empty: {self.data_file}")
                return False
            return True
        except Exception as e:
            print(f"[FATAL] Failed to parse JSON from {self.data_file}: {e}")
            return False

    def print_statistics(self) -> None:
        """Prints statistical overview of the dataset."""
        total_pages = len(self.pages)
        total_chars = sum(p.get("char_count", 0) for p in self.pages)
        total_words = sum(p.get("word_count", 0) for p in self.pages)

        words_per_page = [p.get("word_count", 0) for p in self.pages]
        min_words = min(words_per_page) if words_per_page else 0
        max_words = max(words_per_page) if words_per_page else 0
        avg_words = total_words / total_pages if total_pages else 0

        # Category distribution
        categories: Dict[str, int] = {}
        for p in self.pages:
            cat = p.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

        print("=" * 80)
        print("CarePal Scraped Dataset Statistics (Phase 1)")
        print("=" * 80)
        print(f"Total Pages Scraped    : {total_pages}")
        print(f"Total Word Count       : {total_words:,} words")
        print(f"Total Character Count  : {total_chars:,} characters")
        print(f"Average Words/Page     : {avg_words:.1f} words")
        print(f"Word Count Range       : {min_words} to {max_words:,} words")
        print("-" * 80)
        print("Category Breakdown:")
        for cat, count in sorted(categories.items()):
            print(f"  • {cat.capitalize():<15}: {count:>2} pages")
        print("-" * 80)
        print("Page Details Table:")
        header = f"{'#':<3} | {'Category':<8} | {'Words':>6} | {'Chars':>7} | {'Title':<48}"
        print(header)
        print("-" * len(header))
        for idx, p in enumerate(self.pages, start=1):
            title = p.get("title", "Untitled")[:48]
            cat = p.get("category", "")[:8]
            words = p.get("word_count", 0)
            chars = p.get("char_count", 0)
            print(f"{idx:<3} | {cat:<8} | {words:>6,} | {chars:>7,} | {title:<48}")
        print("=" * 80)

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Runs quality validation battery.
        Returns: (passed: bool, issues: list, warnings: list)
        """
        self.issues = []
        self.warnings = []

        print("\nExecuting Quality & Sanitization Checks...")
        print("-" * 80)

        # Check 1: Empty or minimal React shells
        empty_shell_count = 0
        for p in self.pages:
            url = p.get("url", "")
            content = p.get("content", "")
            chars = p.get("char_count", 0)

            if chars < 150:
                self.issues.append(f"Empty or minimal shell detected ({chars} chars) at {url}")
                empty_shell_count += 1
            if '<div id="root">' in content or "enable JavaScript" in content:
                self.issues.append(f"Unrendered React placeholder detected at {url}")
                empty_shell_count += 1

        if empty_shell_count == 0:
            print("  [PASS] Empty React Shell Check: No empty shells or unrendered placeholders.")
        else:
            print(f"  [FAIL] Empty React Shell Check: {empty_shell_count} issues found.")

        # Check 2: Repeated navbar & footer boilerplate leakage
        boilerplate_leaks = 0
        for p in self.pages:
            url = p.get("url", "")
            content = p.get("content", "")

            for pattern in NAVBAR_BOILERPLATE_PATTERNS:
                if pattern in content:
                    self.issues.append(f"Boilerplate pattern '{pattern}' leaked into {url}")
                    boilerplate_leaks += 1

        if boilerplate_leaks == 0:
            print("  [PASS] Boilerplate Leakage Check: 0 navbar/footer/button strings found.")
        else:
            print(f"  [FAIL] Boilerplate Leakage Check: {boilerplate_leaks} leaks detected.")

        # Check 3: Content Completeness (Required business pages)
        scraped_urls = [p.get("url", "") for p in self.pages]
        missing_core = []
        for req in REQUIRED_CORE_PAGES:
            if not any(req in u for u in scraped_urls):
                missing_core.append(req)

        if not missing_core:
            print("  [PASS] Core Pages Check: All essential services, refund, terms, and privacy pages present.")
        else:
            self.issues.append(f"Missing core business pages: {', '.join(missing_core)}")
            print(f"  [FAIL] Core Pages Check: Missing {len(missing_core)} required pages.")

        # Check 4: Duplicate Content Detection
        seen_hashes: Dict[str, str] = {}
        duplicates = 0
        for p in self.pages:
            url = p.get("url", "")
            content = p.get("content", "")
            content_hash = hashlib.md5(content.strip().encode("utf-8")).hexdigest()

            if content_hash in seen_hashes:
                self.warnings.append(f"Potential duplicate content: {url} matches {seen_hashes[content_hash]}")
                duplicates += 1
            else:
                seen_hashes[content_hash] = url

        if duplicates == 0:
            print("  [PASS] Duplicate Content Check: All pages have distinct content.")
        else:
            print(f"  [WARN] Duplicate Content Check: {duplicates} duplicate/identical pages flagged.")

        # Check 5: Metadata Validity (Title, URL, Scraped timestamp)
        metadata_errors = 0
        for p in self.pages:
            url = p.get("url", "")
            title = p.get("title", "")
            scraped_at = p.get("scraped_at", "")

            if not url or not url.startswith("http"):
                self.issues.append(f"Invalid URL in page metadata: {url}")
                metadata_errors += 1
            if not title or len(title.strip()) < 3:
                self.issues.append(f"Missing or suspicious title at {url}: '{title}'")
                metadata_errors += 1
            if not scraped_at:
                self.warnings.append(f"Missing scraped_at timestamp at {url}")

        if metadata_errors == 0:
            print("  [PASS] Metadata Validity Check: All pages have valid URLs, titles, and timestamps.")
        else:
            print(f"  [FAIL] Metadata Validity Check: {metadata_errors} errors found.")

        print("-" * 80)

        # Summary
        all_passed = len(self.issues) == 0
        if all_passed:
            print(">>> VALIDATION RESULT: ALL CHECKS PASSED (Clean Quality Grade: A+)")
        else:
            print(f">>> VALIDATION RESULT: FAILED ({len(self.issues)} critical issues, {len(self.warnings)} warnings)")
            for issue in self.issues:
                print(f"    • [ISSUE] {issue}")
            for warn in self.warnings:
                print(f"    • [WARNING] {warn}")

        print("=" * 80)
        return all_passed, self.issues, self.warnings


def main():
    inspector = DataInspector()
    if not inspector.load_data():
        sys.exit(1)

    inspector.print_statistics()
    passed, issues, warnings = inspector.validate()

    if not passed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
