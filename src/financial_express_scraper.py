"""
Web Scraper for Financial Express (Business & Market News)
Scrapes all articles from Sep 1, 2025 to Feb 28, 2026.

This is a merged, highly optimized version of the previous scrapers.
It uses ThreadPoolExecutor to scrape concurrently at high speed while
strictly preserving the original sequence of articles on the page.

Environment requirements as per contribution_rule.md:
- Run inside the 'ml' conda environment.
- Install dependencies using: `conda install requests beautifulsoup4`

Instructions for execution:
1. conda activate ml
2. python financial_express_merged_scraper.py
"""

import csv
import json
import os
import re
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# ── Configuration ──────────────────────────────────────────────────────────────
SECTIONS = [
    'https://www.financialexpress.com/business/page/{page_num}/',
    'https://www.financialexpress.com/market/page/{page_num}/'
]
CSV_FILE = 'financial_express_merged_news.csv'
FIELDS = ['date', 'title', 'news', 'url', 'section', 'author']

# Highly Optimized Settings
MAX_WORKERS = 20          # 20 concurrent threads for blazing-fast IO scraping
DELAY_BETWEEN_PAGES = 0.5 # Minimal delay between listing pages
MAX_RETRIES = 3

# Date range: Sep 1, 2025 to Feb 28, 2026 (inclusive)
START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2026, 2, 28, 23, 59, 59)


def get_article_urls(page_url):
    """Fetches a listing page and extracts all article URLs in sequence."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                page_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
                timeout=20
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            links = []
            seen = set()
            # Non-article paths to exclude
            exclude_patterns = [
                '/page/', '/latest-news/', '/about/', '/shorts/',
                '/author/', '/related-news/', '/subscribe'
            ]
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Match any financialexpress.com article URL ending with a numeric ID.
                if ('financialexpress.com/' in href
                        and re.search(r'[-/]\d+/?$', href)
                        and not any(excl in href for excl in exclude_patterns)
                        and href not in seen):
                    links.append(href)
                    seen.add(href)

            return links

        except Exception as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Error fetching {page_url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    return []


def scrape_article(url):
    """Scrapes date, title, news body, author from a single article URL."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
                timeout=20
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Determine section from URL for better organization
            section = 'Business' if '/business/' in url else ('Market' if '/market/' in url else 'Other')

            # Method 1: JSON-LD
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string, strict=False)
                    if isinstance(data, list):
                        for item in data:
                            if item.get('@type') in ['NewsArticle', 'Article', 'WebPage']:
                                data = item
                                break
                    elif isinstance(data, dict) and '@graph' in data:
                        for item in data['@graph']:
                            if item.get('@type') in ['NewsArticle', 'Article', 'WebPage']:
                                data = item
                                break

                    if isinstance(data, dict) and data.get('@type') in ['NewsArticle', 'Article']:
                        title = data.get('headline', '')
                        date = data.get('datePublished', '')
                        news = data.get('articleBody', '')

                        author_data = data.get('author')
                        if isinstance(author_data, dict):
                            author = author_data.get('name', '')
                        elif isinstance(author_data, list) and len(author_data) > 0:
                            author = author_data[0].get('name', '')
                        else:
                            author = str(author_data or '')

                        if title and news:
                            return {
                                'date': date,
                                'title': title,
                                'news': news.strip(),
                                'url': url,
                                'section': section,
                                'author': author
                            }
                except Exception:
                    pass

            # Method 2: HTML Parsing Fallback
            title_tag = soup.find('h1')
            title = title_tag.text.strip() if title_tag else (soup.title.string if soup.title else '')

            date_meta = soup.find('meta', property='article:published_time')
            date = date_meta['content'] if date_meta else ''
            if not date:
                time_tag = soup.find('time')
                date = time_tag.text.strip() if time_tag else ''

            author_meta = soup.find('meta', attrs={'name': 'author'})
            author = author_meta['content'] if author_meta else ''

            paragraphs = soup.find_all('p')
            news_text = '\n'.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 30])

            return {
                'date': date,
                'title': title,
                'news': news_text,
                'url': url,
                'section': section,
                'author': author
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            else:
                pass # Silent fail after max retries
    return None


def parse_article_date(date_str):
    """Parses an ISO-8601 date string and returns a naive datetime object."""
    if not date_str:
        return None
    try:
        clean = re.sub(r'[+-]\d{2}:\d{2}$', '', date_str)
        return datetime.fromisoformat(clean)
    except Exception:
        return None


def append_to_csv(csv_path, rows, write_header=False):
    """Appends rows to the CSV file. Writes header if the file is new."""
    mode = 'w' if write_header else 'a'
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 80)
    print("Financial Express Merged & Optimized Scraper")
    print(f"Date range: {START_DATE.strftime('%b %d, %Y')} → {END_DATE.strftime('%b %d, %Y')}")
    print(f"Threads: {MAX_WORKERS} (High Speed)")
    print("=" * 80)

    # Dictionary to track which page each section left off on
    section_starts = {sec: 1 for sec in SECTIONS}
    file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
    
    if file_exists:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f) - 1
        print(f"Existing CSV found with {existing_count} articles.")
        resume = input("Resume scraping and APPEND? (y/n): ").strip().lower()
        if resume == 'y':
            for sec in SECTIONS:
                name = 'Business' if '/business/' in sec else 'Market'
                start_pg = input(f"Enter the starting page number to resume for section '{name}': ").strip()
                section_starts[sec] = int(start_pg) if start_pg.isdigit() else 1
        else:
            file_exists = False
            print("Starting fresh — existing CSV will be overwritten.")

    if not file_exists:
        append_to_csv(CSV_FILE, [], write_header=True)

    global_seen_urls = set()
    total_saved = 0

    # Share the thread pool across everything for maximum efficiency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        for section_url in SECTIONS:
            section_name = 'Business' if '/business/' in section_url else 'Market'
            page_num = section_starts[section_url]
            reached_start_date = False
            
            print(f"\n{'=' * 80}")
            print(f"Starting section: {section_name}")
            print(f"{'=' * 80}")
            
            while not reached_start_date:
                page_url = section_url.format(page_num=page_num)
                print(f"\n{'─' * 60}")
                print(f"Page {page_num}: {page_url}")

                article_urls = get_article_urls(page_url)

                if not article_urls:
                    print(f"  No articles found on this page. Section complete.")
                    break

                new_urls = [u for u in article_urls if u not in global_seen_urls]
                for u in new_urls:
                    global_seen_urls.add(u)

                if not new_urls:
                    print(f"  All articles on page {page_num} already scraped. Skipping.")
                    page_num += 1
                    time.sleep(DELAY_BETWEEN_PAGES)
                    continue

                print(f"  Scraping {len(new_urls)} articles using {MAX_WORKERS} concurrent threads...")

                # CRITICAL: executor.map guarantees results are yielded in the EXACT SAME ORDER as new_urls!
                results = list(executor.map(scrape_article, new_urls))

                page_articles = []
                oldest_date_on_page = None

                for data in results:
                    if not data or not data['title'] or not data['news']:
                        continue

                    dt = parse_article_date(data['date'])

                    if dt and (oldest_date_on_page is None or dt < oldest_date_on_page):
                        oldest_date_on_page = dt

                    if dt is None:
                        page_articles.append(data)
                    elif START_DATE <= dt <= END_DATE:
                        page_articles.append(data)

                if page_articles:
                    append_to_csv(CSV_FILE, page_articles)
                    total_saved += len(page_articles)
                    print(f"  ✓ Saved {len(page_articles)} articles (Total so far: {total_saved})")
                else:
                    print(f"  No articles in date range on this page.")

                if oldest_date_on_page:
                    print(f"  Oldest article on page: {oldest_date_on_page.strftime('%b %d, %Y')}")
                    if oldest_date_on_page < START_DATE:
                        print(f"\n  ✓ Reached articles older than {START_DATE.strftime('%b %d, %Y')}. Stopping {section_name}.")
                        reached_start_date = True
                else:
                    print(f"  Warning: Could not determine dates on page {page_num}.")

                page_num += 1
                if not reached_start_date:
                    time.sleep(DELAY_BETWEEN_PAGES)

    print(f"\n{'=' * 80}")
    print(f"Scraping complete! Total articles saved: {total_saved}")
    print(f"Combined data saved to: {CSV_FILE}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
