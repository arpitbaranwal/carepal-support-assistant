"""
Web Scraper Module for TheCarePal (Phase 1).
Extracts public pages from https://www.thecarepal.com, renders dynamic React SPA
components (via Playwright or prerendered DOM), and rigorously strips navigation,
footer, sidebar TOC, and floating button boilerplate.
"""

import sys
import os
import json
import re
import time
import unicodedata
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

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


class ScrapedPage:
    """Represents a clean, structured scraped page."""

    def __init__(
        self,
        url: str,
        title: str,
        meta_description: str,
        category: str,
        label: str,
        main_heading: str,
        content: str,
        scraped_at: Optional[str] = None,
    ):
        self.url = url
        self.title = self._clean_text(title)
        self.meta_description = self._clean_text(meta_description)
        self.category = category
        self.label = label
        self.main_heading = self._clean_text(main_heading)
        self.content = self._clean_text(content).strip()
        self.scraped_at = scraped_at or datetime.now(timezone.utc).isoformat()
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize Unicode and clean anomalous whitespace."""
        if not text:
            return ""
        # Normalize NFKC Unicode
        text = unicodedata.normalize("NFKC", text)
        # Fix common mojibake characters
        replacements = {
            "â€“": "–",
            "â€”": "—",
            "â€™": "'",
            "â€˜": "'",
            "â€œ": '"',
            "â€": '"',
            "\u00a0": " ",
            "\u200b": "",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "title": self.title,
            "meta_description": self.meta_description,
            "category": self.category,
            "label": self.label,
            "main_heading": self.main_heading,
            "content": self.content,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "scraped_at": self.scraped_at,
        }


class CarePalScraper:
    """Scraper tailored for TheCarePal React SPA platform."""

    def __init__(self, use_browser: bool = False, timeout: int = config.REQUEST_TIMEOUT):
        self.use_browser = use_browser
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _fetch_html_http(self, url: str) -> Optional[str]:
        """Fetch raw HTML using requests session."""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                return resp.text
            else:
                print(f"  [HTTP {resp.status_code}] Failed to fetch {url}")
                return None
        except Exception as e:
            print(f"  [Error] HTTP fetch error for {url}: {e}")
            return None

    def _fetch_html_playwright(self, url: str) -> Optional[str]:
        """Render React SPA using Playwright headless Chromium."""
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({"User-Agent": config.USER_AGENT})
                page.goto(url, timeout=self.timeout * 1000, wait_until="networkidle")
                # Wait for React root to mount children
                page.wait_for_selector("#root", timeout=5000)
                html = page.content()
                browser.close()
                return html
        except Exception as e:
            print(f"  [Error] Playwright render failed for {url}: {e}")
            return None

    def _strip_boilerplate(self, soup: BeautifulSoup) -> None:
        """Removes navbars, footers, floating buttons, aside sidebars, scripts, styles."""
        # 1. Decompose unwanted tags
        for tag_name in config.BOILERPLATE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 2. Decompose aside navigation (e.g. terms table-of-contents sidebar)
        for aside in soup.find_all("aside"):
            aside.decompose()

        # 3. Decompose elements matching boilerplate class names
        for cls_name in config.BOILERPLATE_CLASSES:
            for element in soup.find_all(class_=lambda x: x and cls_name in str(x)):
                element.decompose()

        # 4. Decompose known boilerplate IDs
        for element in soup.find_all(id=lambda x: x and any(b in str(x).lower() for b in ["navbar", "footer", "chat-widget"])):
            element.decompose()

    def _extract_clean_text(self, soup: BeautifulSoup) -> Tuple[str, str, str, str]:
        """
        Extracts title, meta description, main heading, and formatted content.
        Preserves headers, paragraphs, and list items without duplication.
        """
        # Title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Meta description
        meta_desc = ""
        desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if desc_tag and desc_tag.get("content"):
            meta_desc = desc_tag["content"].strip()

        # Main heading (h1)
        h1_tag = soup.find("h1")
        main_heading = h1_tag.get_text(strip=True) if h1_tag else ""

        # Content container: prioritize top-level <main> tags to avoid nested duplicate text
        all_mains = soup.find_all("main")
        if all_mains:
            # Keep only outermost <main> elements (filter out children of other <main> tags)
            top_mains = [m for m in all_mains if not any(p in all_mains for p in m.parents)]
            text_blocks = [self._parse_container_blocks(m) for m in top_mains]
            full_text = "\n\n".join(t for t in text_blocks if t.strip())
        else:
            # Fallback to body
            container = soup.find("body") or soup
            full_text = self._parse_container_blocks(container)

        # Clean noise phrases and duplicate CTA links
        cleaned_lines = []
        for line in full_text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            # Remove exact boilerplate noise lines
            if line_str in (
                "Book an Appointment",
                "Talk to an Expert",
                "Click here to call",
                "🗓️ Book Now",
                "Schedule Your Consultation",
                "Your browser does not support the video tag.",
                "×",
            ):
                continue

            # Filter out lone button icons or micro artifacts
            if len(line_str) == 1 and not line_str.isalnum():
                continue

            cleaned_lines.append(line_str)

        final_content = "\n".join(cleaned_lines)
        return title, meta_desc, main_heading, final_content

    def _parse_container_blocks(self, container) -> str:
        """Parse structured elements inside a container preserving logical sections."""
        blocks = []
        for element in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd"]):
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue

            tag = element.name.lower()
            if tag in ["h1", "h2", "h3", "h4"]:
                blocks.append(f"\n## {text}\n")
            elif tag == "li":
                blocks.append(f"• {text}")
            elif tag == "dt":
                blocks.append(f"\nQ: {text}")
            elif tag == "dd":
                blocks.append(f"A: {text}\n")
            else:
                blocks.append(text)

        if not blocks:
            # Fallback to general text extraction
            raw_text = container.get_text(separator="\n", strip=True)
            blocks = [raw_text]

        # Deduplicate consecutive identical blocks
        deduped = []
        for b in blocks:
            if not deduped or deduped[-1] != b:
                deduped.append(b)

        return "\n".join(deduped)

    def scrape_url(self, target_info: Dict) -> Optional[ScrapedPage]:
        """Scrapes a single URL and returns a ScrapedPage object."""
        url = target_info["url"]
        category = target_info.get("category", "general")
        label = target_info.get("label", "")

        html = None
        if self.use_browser:
            html = self._fetch_html_playwright(url)
        else:
            html = self._fetch_html_http(url)
            # Automatic fallback: if HTTP returns empty React root shell, use Playwright
            if html and '<div id="root"></div>' in html and len(html) < 2000:
                print(f"  [SPA Fallback] Empty React root detected on {url}. Launching Playwright...")
                html = self._fetch_html_playwright(url)

        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")
        self._strip_boilerplate(soup)
        title, meta_desc, heading, content = self._extract_clean_text(soup)

        if not content or len(content) < 80:
            print(f"  [Warning] Insufficient content extracted ({len(content)} chars) from {url}")
            return None

        return ScrapedPage(
            url=url,
            title=title or label or "The CarePal",
            meta_description=meta_desc or target_info.get("description", ""),
            category=category,
            label=label,
            main_heading=heading,
            content=content,
        )

    def discover_sitemap_urls(self) -> List[Dict]:
        """Fetches all URLs listed in TheCarePal's sitemap.xml."""
        print(f"Fetching sitemap from {config.SITEMAP_URL}...")
        try:
            resp = self.session.get(config.SITEMAP_URL, timeout=self.timeout)
            if resp.status_code != 200:
                print(f"Failed to fetch sitemap: HTTP {resp.status_code}")
                return config.CORE_TARGET_PAGES

            root = ET.fromstring(resp.content)
            urls = []
            for elem in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                loc = elem.text.strip()
                # Determine category
                if "/blog/" in loc or loc.endswith("/blogs"):
                    category = "blog"
                elif any(p in loc for p in ["terms", "privacy", "refund"]):
                    category = "policy"
                elif any(p in loc for p in ["elderly", "dementia", "alzeihmer", "livein", "autism", "baby", "therapy", "doctor", "emergency", "pharmaceutical", "service"]):
                    category = "service"
                elif any(p in loc for p in ["contact", "careers"]):
                    category = "support"
                else:
                    category = "company"

                slug = loc.rstrip("/").split("/")[-1]
                label = slug.replace("-", " ").title() if slug else "Home"

                urls.append({
                    "url": loc,
                    "category": category,
                    "label": label,
                    "description": f"Content from {loc}",
                })
            print(f"Discovered {len(urls)} URLs from sitemap.xml.")
            return urls
        except Exception as e:
            print(f"Error parsing sitemap: {e}. Falling back to CORE_TARGET_PAGES.")
            return config.CORE_TARGET_PAGES

    def run(self, urls: Optional[List[Dict]] = None, output_file: Optional[Path] = None) -> List[Dict]:
        """Runs scraping pipeline over target URLs and writes to JSON."""
        targets = urls or config.CORE_TARGET_PAGES
        output_path = output_file or config.RAW_PAGES_FILE

        print("=" * 70)
        print("CarePal Web Scraper Pipeline (Phase 1)")
        print(f"Target count: {len(targets)} pages")
        print(f"Browser mode: {'Playwright Headless' if self.use_browser else 'Fast HTTP + Prerender'}")
        print(f"Output destination: {output_path}")
        print("=" * 70)

        results = []
        failed = []

        start_time = time.time()
        for idx, target in enumerate(targets, start=1):
            url = target["url"]
            print(f"[{idx}/{len(targets)}] Scraping: {url} ({target.get('category', '')})")
            page = self.scrape_url(target)
            if page:
                results.append(page.to_dict())
                print(f"  ✓ Success: {page.char_count} chars, {page.word_count} words | Title: {page.title[:50]}...")
            else:
                failed.append(url)
                print(f"  ✗ Failed: {url}")

        elapsed = time.time() - start_time

        # Save to raw_pages.json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("Scraping Summary")
        print(f"Successfully scraped : {len(results)}/{len(targets)} pages")
        print(f"Failed pages         : {len(failed)}")
        print(f"Total time elapsed   : {elapsed:.2f} seconds")
        print(f"Data saved to        : {output_path}")
        print("=" * 70)

        return results


def main():
    parser = argparse.ArgumentParser(description="Scrape TheCarePal public website pages.")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Use Playwright headless browser for full React rendering",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all pages from sitemap.xml including blogs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output JSON path",
    )
    args = parser.parse_args()

    use_browser = args.browser or config.USE_HEADLESS_BROWSER
    scraper = CarePalScraper(use_browser=use_browser)

    if args.all:
        target_urls = scraper.discover_sitemap_urls()
    else:
        target_urls = config.CORE_TARGET_PAGES

    out_file = Path(args.output) if args.output else config.RAW_PAGES_FILE
    scraper.run(urls=target_urls, output_file=out_file)


if __name__ == "__main__":
    main()
