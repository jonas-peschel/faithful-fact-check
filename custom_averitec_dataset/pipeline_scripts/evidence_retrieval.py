import argparse
import os 
from dotenv import load_dotenv  
import time
from time import sleep
import json
from tqdm.auto import tqdm 
from urllib.parse import urlparse
from dateutil import parser
import trafilatura
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError 
import requests
from pathlib import Path 
from utils import load_json, save_json 

# list of websites forbidden for google search (fact-checking websites)
FORBIDDEN_WEBSITES = [
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "washingtonpost.com/news/fact-checker",
    "apnews.com/hub/ap-fact-check",
    "fullfact.org",
    "reuters.com/fact-check",
    "huggingface.co"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Web-based evidence retrieval using Serper API and trafilatura for web scraping.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the AVeriTeC data file.")
    parser.add_argument("--store_folder", type=str, default="web_evidence", help="Path to the folder where the scraped web page content gets stored")
    parser.add_argument("--n_pages", type=int, default=3, help="Number of pages per search query for Google search")
    parser.add_argument("--start_idx", type=int, default=0, help="Claim to start with")
    parser.add_argument("--end_idx", type=int, default=-1, help="Claim to end with")
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

    search_results = []
    for page in range(1, n_pages+1):
        search_params = {
            "q": query,
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
            if (domain in FORBIDDEN_WEBSITES):
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

def scrape_page(url, trafilatura_config):
    """Fetch and scrape a URL using trafilatura package.
    Return textual content and potential error.
    """

    # fetch URL
    n_tries = 1     # more tries do not seem to help 
    for i in range(1, n_tries+1):
        try:
            fetched = trafilatura.fetch_url(url, config=trafilatura_config)
            if fetched is None:
                return None, "Fetch failed"

            content = trafilatura.extract(fetched, include_comments=False, with_metadata=False, no_fallback=False, config=trafilatura_config)
            if not content:
                return None, "No content extracted"

            return content, None 
        except Exception as e:
            if i == n_tries:
                return None, str(e)
            sleep(3)

def parallel_scraping(urls, search_infos, trafilatura_config, max_workers=4, max_pages=None):

    successful_results = []
    failed_results = []
    TIMEOUT = 300 # wait up to 5 min for one page

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scrape_page, url, trafilatura_config): (url, info) for url, info in zip(urls, search_infos)}   # submit all scraping tasks

        with tqdm(total=len(urls), desc="Scrape web pages", leave=False) as prog_bar:
            for future in as_completed(futures):
                url, search_info = futures[future]

                # get result from the submitted future
                try:
                    content, error = future.result(timeout=TIMEOUT)

                    if not error:
                        successful_results.append({
                            "url": url,
                            "search_info": search_info,
                            "content": content
                        })
                        print(f"Successfully scraped URL: {url}")

                        # check for early stopping condition
                        if ((max_pages is not None) and (len(successful_results) >= max_pages)):
                            print("Reached early stopping condition!")
                            # cancel remaining futures
                            for future in futures:
                                if not future.done():
                                    future.cancel()
                            break
                    else:
                        failed_results.append({
                            "url": url,
                            "search_info": search_info,
                            "error": error
                        })
                        print(f"Failed scraping URL: {url} with error: {error}")
                except TimeoutError:
                    error = f"Failed scraping URL within the timeout limit of {TIMEOUT}s."
                    failed_results.append({
                            "url": url,
                            "search_info": search_info,
                            "error": error
                        })
                    print(f"Failed scraping URL: {url} with error: {error}")
                except Exception as e:
                    error = str(e)
                    failed_results.append({
                            "url": url,
                            "search_info": search_info,
                            "error": error
                        })
                    print(f"Failed scraping URL: {url} with error: {error}")
                prog_bar.update(1)

    return successful_results, failed_results 

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
        # 1. get URLs from Google search via Serper API
        urls, search_infos = get_all_urls(claim_results, config.n_pages, SERPER_API_KEY)

        # 2. scrape URLs using trafilatura
        successful_results, failed_results = parallel_scraping(urls, search_infos, trafilatura_config, max_workers=config.n_scrape_workers, max_pages=config.max_pages)

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