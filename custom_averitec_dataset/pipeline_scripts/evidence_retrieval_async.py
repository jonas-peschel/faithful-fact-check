import argparse
import os 
import io
import gc 
import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from dotenv import load_dotenv  
import time
from time import sleep
import json
from tqdm.auto import tqdm 
from urllib.parse import urlparse
from dateutil import parser
import trafilatura
from playwright.async_api import async_playwright 
from playwright_stealth import Stealth
import pdfplumber
import requests
from pathlib import Path 
from utils import load_json, save_json 
import random 
import re 

# list of websites forbidden for google search (fact-checking websites)
FORBIDDEN_DOMAINS = [
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "fullfact.org",
    "huggingface.co",
    "facebook.com",
]

# different header profiles for web scraping
HEADER_PROFILES = [
    {
        # Chrome, Windows, English
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
    },
    {
        # Firefox, Windows, English/French
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.8,fr;q=0.6",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.bing.com/",
    },
    {
        # Chrome, macOS, German
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.ecosia.org/",
    },
]

def parse_args():
    parser = argparse.ArgumentParser(description="Web-based evidence retrieval using Serper API and trafilatura for web scraping.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the AVeriTeC data file.")
    parser.add_argument("--store_folder", type=str, default="web_evidence", help="Path to the folder where the scraped web page content gets stored")
    parser.add_argument("--n_pages", type=int, default=3, help="Number of pages per search query for Google search")
    parser.add_argument("--start_idx", type=int, default=0, help="Claim to start with")
    parser.add_argument("--end_idx", type=int, default=None, help="Claim to end with")
    parser.add_argument("--trafilatura_config", type=str, default="retrieval_configs/config.cfg", help="Path to the trafilatura config file")
    parser.add_argument("--n_scrape_workers", type=int, default=6, help="Number of workers for web scraping")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to retrieve per claim")

    return parser.parse_args()

def get_domain_name(url):

    if '://' not in url:
        url = 'http://' + url
    domain = urlparse(url).netloc

    # remove leading "www."
    domain = domain.replace("www.", "")
    return domain

def format_claim_date(claim):
    """Format the claim's date for restricted Google search"""
    
    try:
        day, month, year = claim["claim_date"].split("-")
    except:
        year, month, day = "2022", "01", "01"

    if len(year) == 2 and int(year) <= 26:
        year = "20" + year
    elif len(year) == 2:
        year = "19" + year
    elif len(year) == 1:
        year = "200" + year
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    claim_date = "/".join([month, day, year])

    return claim_date 

def setup_search_strings(claim_results):
    """
    Search strings for a claim:
    1. Raw claim text
    2. Raw claim text + author
    3. Search queries (generated in previous pipeline step)
    """

    search_strings = []
    search_types = []

    # 1. raw claim text
    search_strings.append(claim_results["claim"])
    search_types.append("raw-claim")

    # 2. claim text + author 
    speaker = claim_results["speaker"].strip() if claim_results["speaker"] else None 
    if speaker:
        search_strings.append(speaker + " " + claim_results["claim"])
        search_types.append("claim+author")

    # 3. search queries
    search_queries = claim_results["search_queries"]
    search_strings.extend(search_queries)
    search_types.extend(["search_query"] * len(search_queries))

    return search_strings, search_types

def date_filter(search_results, claim_date):
    """
    Filter the search results by date. Keep search results only if they
    have a date prior to the claim's date.
    """

    def compare_dates(search_result, claim_date):
        """Return True if result's date is earlier than claim's date"""

        # try to get date of search result, if not defined return True by default
        result_date = search_result.get("date", None)
        if not result_date:
            return True 

        # since the format of the claim's date seems to be not standardized
        # we try parsing the date and if it fails we return True by default
        try:
            result_dt = parser.parse(result_date)
            claim_dt = parser.parse(claim_date)
            return claim_dt > result_dt
        except Exception as e:
            print(str(e))
            return True
        
    filtered_search_results = [search_result for search_result in search_results if compare_dates(search_result, claim_date)]
    return filtered_search_results 

def google_search(query, api_key, claim_date, n_pages, is_date_filtering):
    serper_url = "https://google.serper.dev/search"

    # exclude forbidden domains directly in the query to not "waste" results
    exclude_str = " " + " ".join([f"-site:{domain}" for domain in FORBIDDEN_DOMAINS])

    search_results = []
    for page in range(1, n_pages+1):
        search_params = {
            "q": query + exclude_str,
            "autocorrect": False,
            # "num": 10,
            "page": page,
            "gl": "us",
            "hl": "en"
        }
        if is_date_filtering:
            search_params['tbs'] = f"cdr:1,cd_min:01/01/1900,cd_max:{claim_date}"

        payload = json.dumps(search_params)
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        response = requests.request("POST", serper_url, headers=headers, data=payload)
        result = response.json()
        if 'organic' in result:
            search_results.extend(result['organic'])   
    return search_results
    
def get_google_search_results(search_query, api_key, claim_date, n_pages):
    """
    Try google search with incorporated date filtering, if no results are found
    try without, return search result items with (additional) post-hoc date filtering.
    """

    n_tries = 3
    search_results_full = []
    for i_try in range(n_tries):
        try:
            search_results = google_search(query = search_query,
                                            api_key = api_key,
                                            claim_date = claim_date,
                                            n_pages = n_pages,
                                            is_date_filtering = True)
            
            if len(search_results) == 0:
                break
            
            search_results = date_filter(search_results, claim_date)
            search_results_full.extend(search_results)
        except Exception:
            sleep(3 + 2*i_try)

    # check if a sufficient number of web pages was retrieved; if not fall back to post-hoc date filtering
    if len(search_results_full) >= 10:
        return search_results_full 

    print("Not enough search results found for this search query. Falling back to post-hoc date filtering...")
    for i_try in range(n_tries):
        try:
            search_results = google_search(query = search_query,
                                           api_key = api_key,
                                           claim_date = claim_date,
                                           n_pages = n_pages,
                                           is_date_filtering = False)

            search_results = date_filter(search_results, claim_date)
            search_results_full.extend(search_results)     
        except Exception:
            sleep(3 + 2*i_try)

    return search_results_full 

def get_all_urls(claim_results, n_pages, api_key):

    urls = []
    search_infos = []
    search_strings, search_types = setup_search_strings(claim_results)

    for search_string, search_type in tqdm(list(zip(search_strings, search_types)), leave=False, desc="Google Search Queries"):
        search_results = get_google_search_results(search_query=search_string, 
                                                   api_key=api_key, 
                                                   claim_date=format_claim_date(claim_results), 
                                                   n_pages=n_pages) 
        
        # loop through search result items and store them if they are new
        for search_result in search_results:
            url = str(search_result['link'])
            domain = get_domain_name(url)
            title = search_result.get("title", None)
            date = search_result.get("date", None)

            # skip search results from forbidden sites
            if (domain in FORBIDDEN_DOMAINS):
                continue

            if url not in urls:
                urls.append(url)
                search_infos.append({
                    "url": url,
                    "search_string": search_string,
                    "search_type": search_type,
                    "title": title,
                    "date": date
                })
    return urls, search_infos

def scrape_pdf(url):
    """Scrape PDF file"""

    def line_is_noise(line_text):
        if line_text.isdigit():
            return True 
        if((len(line_text) <= 5) and not any(char.isalpha() for char in line_text)):
            return True 
        return False 

    def line_is_reoccurring(line_text, lines_dict):
        """Remove lines re-occurring many times, e.g., headers."""
        # for a line to be removed it must re-occurr multiple times AND have a meaningful length
        # i.e., extremely short lines can plausibly re-occurr and should not be removed here 
        if len(line_text) >= 10:
            if lines_dict[line_text] >= 5:
                return True 
        return False

    try:
        headers = random.choice(HEADER_PROFILES)
        response = requests.get(url, headers=headers, timeout=30) 
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code} when fetching PDF"
        
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            lines = []
            lines_dict = {}
            for page in pdf.pages:
                words = page.extract_words(use_text_flow=True, split_at_punctuation=False)
                if not words:
                    continue

                page_lines = []
                current_line = [words[0]]
                for word in words[1:]:
                    prev_word = current_line[-1]
                    # group words into lines based on vertical position
                    if abs(word["top"] - prev_word["top"]) < 3:  # use 3 as heuristic
                        current_line.append(word)
                    else:
                        page_lines.append(current_line)
                        # keep track of line occurrences
                        current_line_text = " ".join(w["text"] for w in current_line)
                        lines_dict[current_line_text] = lines_dict.get(current_line_text, 0) + 1
                        current_line = [word]

                page_lines.append(current_line)
                # keep track of line occurrences
                current_line_text = " ".join(w["text"] for w in current_line)
                lines_dict[current_line_text] = lines_dict.get(current_line_text, 0) + 1
                lines.append(page_lines)

        paragraphs = []
        for page_lines in lines:
            # compute typical line spacing (for this page)
            if len(page_lines) > 1:
                spacings = [
                    page_lines[i+1][0]["top"] - page_lines[i][0]["top"]
                    for i in range(len(page_lines) - 1)
                ]
                median_spacing = sorted(spacings)[len(spacings) // 2]
                paragraph_threshold = median_spacing * 1.5
            else:
                paragraph_threshold = 20  # fallback

            # join lines into paragraphs based on spacing
            page_paragraphs = []
            first_line_text = " ".join(w["text"] for w in page_lines[0])
            current_paragraph = [first_line_text] if (not line_is_noise(first_line_text.strip()) 
                                                      and not line_is_reoccurring(first_line_text, lines_dict)) else []
            for i in range(1, len(page_lines)):
                spacing = page_lines[i][0]["top"] - page_lines[i-1][0]["top"]
                line_text = " ".join(w["text"] for w in page_lines[i])
                # remove non-content lines, e.g., page numbers
                if line_is_noise(line_text.strip()) or line_is_reoccurring(line_text, lines_dict):
                    continue
                if spacing > paragraph_threshold:
                    page_paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = [line_text]
                else:
                    current_paragraph.append(line_text)
            page_paragraphs.append(" ".join(current_paragraph))

            # merge with last paragraph from the previous page if the paragraph spans across pages
            if paragraphs and page_paragraphs:
                last = paragraphs[-1]
                if not last.strip()[-1] in ".!?:":
                    paragraphs[-1] = last + " " + page_paragraphs[0]
                    paragraphs.extend(page_paragraphs[1:])
                else:
                    paragraphs.extend(page_paragraphs)
            else:
                paragraphs.extend(page_paragraphs)

        # remove extremely short paragraphs 
        paragraphs = [p for p in paragraphs if len(p) > 5]

        for i, p in enumerate(paragraphs):
            # clean up multiple consecutive spaces and newlines
            p = re.sub(r"\s+", " ", p)  
            paragraphs[i] = p

        content = "\n\n".join(paragraphs)

        if content:
            return content, None 
        else:
            return None, "No content extracted from PDF"

    except Exception as e:
        return None, str(e)
    finally:
        if response:
            response.close()
        gc.collect()
    
async def scrape_html(browser, url, trafilatura_config):
    """Scrape textual content from HTML page using Playwright + trafilatura"""

    try:
        ## first option: use trafilatura to fetch the html
        # 1. fetch raw html
        html = await asyncio.to_thread(trafilatura.fetch_url, url, config=trafilatura_config)

        # 2. extract content from raw html
        content = await asyncio.to_thread(
            trafilatura.extract,
            html, 
            include_comments=False, 
            with_metadata=False, 
            no_fallback=False, 
            config=trafilatura_config,
        )
        if content: 
            return content, None 

        ## fallback option: Playwright
        print("Trafilatura failed to fetch URL or no content was extracted. Falling back to Playwright...")
        profile = random.choice(HEADER_PROFILES) 
        context = await browser.new_context(
            user_agent=profile["User-Agent"],
            extra_http_headers={k: v for k, v in profile.items() if k != "User-Agent"}
        )
        page = await context.new_page()
        stealth = await Stealth()
        await stealth.apply_stealth_async(page)

        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000) 
        await asyncio.sleep(1)  # Wait for JS to finish
        if not response or response.status != 200:
            return None, f"HTTP error {response.status if response else ''}"
        
        html = await page.content() # Get the full rendered HTML
        await context.close()

        # 2. extract content from raw html
        content = await asyncio.to_thread(
            trafilatura.extract,
            html, 
            include_comments=False, 
            with_metadata=False, 
            no_fallback=False, 
            config=trafilatura_config,
        )
        if not content:
            return None, "No content extracted"

        return content, None 
    except Exception as e:
        return None, str(e)


async def scrape_page(browser, url, trafilatura_config):
    """Fetch and scrape given URL."""

    # check if the URL is a .pdf file
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return await asyncio.to_thread(scrape_pdf, url)
    else:
        return await scrape_html(browser, url, trafilatura_config)
        
async def parallel_scraping(urls, search_infos, trafilatura_config, max_concurrent=6):
        
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def sem_task(url, info):
            async with semaphore:
                content, error = await scrape_page(browser, url, trafilatura_config)
                return (url, info, content, error)
            
        tasks = [sem_task(url, info) for url, info in zip(urls, search_infos)]
        results = await tqdm_async.gather(*tasks, desc="Scrape web pages", leave=False)

        await browser.close()
        return results

def main(config=None):

    if config is None:
        config = parse_args() 

    trafilatura_config = trafilatura.settings.use_config(config.trafilatura_config) # trafilatura config
    
    load_dotenv()
    SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

    results = load_json(config.results_path)
    for idx, claim_results in tqdm(enumerate(results[config.start_idx:config.end_idx], start=config.start_idx), 
                                   desc="Claims", total=len(results[config.start_idx:config.end_idx])):
        
        # measure time for each claim
        start_time = time.time()

        # make folder to store retrieved and scraped content
        Path(config.store_folder + f"/claim_{idx}").mkdir(exist_ok=True, parents=True)

        ## Retrieval
        # 1. get URLs from Google search via SERP API
        urls, search_infos = get_all_urls(claim_results, config.n_pages, SERPER_API_KEY)

        # 2. scrape URLs using trafilatura / Playwright
        results = asyncio.run(
            parallel_scraping(urls, search_infos, trafilatura_config, max_concurrent=config.n_scrape_workers)
        )

        successful_results, failed_results = [], []
        for url, info, content, error in results:
            if not error:
                successful_results.append({
                    "url": url,
                    "search_info": info,
                    "content": content,
                })
                print(f"Successfully scraped URL: {url}")
            else:
                failed_results.append({
                    "url": url,
                    "search_info": info,
                    "error": error,
                })
                print(f"Failed scraping URL: {url} with error: {error}")

        # 3. store the successfully scraped content and corresponding search infos
        for i, search_result in enumerate(successful_results, start=1):

            store_file_path = config.store_folder + f"/claim_{idx}/search_result_{str(i)}.txt"
            with open(store_file_path, "w") as f:
                f.write(search_result['content'])

            # print logging information for each search result
            line = [
                str(idx), 
                claim_results["claim"], 
                search_result['url'], 
                search_result['search_info']['search_string'], 
                search_result['search_info']['search_type'], 
                store_file_path
            ]
            line = "\t".join(line)
            print(line)

        infos = [result['search_info'] for result in successful_results]
        store_file_path_infos = config.store_folder + f"/claim_{idx}/search_infos.json"
        save_json(store_file_path_infos, infos)

        claim_scrape_time = time.time() - start_time
        print(f"\nClaim {idx} processed!")
        print(f"Successfull scraping: {len(successful_results)}")
        print(f"Failed scraping: {len(failed_results)}")
        print(f"Time: {claim_scrape_time:.1f}s")

if __name__ == "__main__":
    main()