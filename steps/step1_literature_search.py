import requests
import pandas as pd
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================================================
# CONFIG
# =========================================================
SEMANTIC_PAGE_SIZE = 50
SEMANTIC_MAX_RESULTS = 400
OPENALEX_MAX_RESULTS = 400
ARXIV_MAX_RESULTS = 300

API_THREADS = 3
PAGE_THREADS = 6

CACHE_FOLDER = "cache"
CACHE_VERSION = "v4_fast_csv_no_alt_pairs"
USER_AGENT = "AutoLiteratureSurvey/1.0 (mailto:test@example.com)"

# =========================================================
# UTILITIES
# =========================================================
def normalize_title(title):
    return re.sub(r"\W+", "", title.lower()) if title else None

def normalize_doi_to_url(doi):
    if not doi:
        return None
    doi = str(doi).lower().strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
    return f"https://doi.org/{doi}" if doi else None

def year_is_valid(year, min_year, max_year):
    return year is None or (min_year <= year <= max_year)

def is_review_paper(title):
    return "YES" if title and "review" in title.lower() else "NO"

def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def get_retry_session():
    session = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session

def get_cache_path(query, min_year, max_year):
    safe = re.sub(r"\W+", "_", query.lower())[:120]
    filename = f"{safe}_{min_year}_{max_year}_{CACHE_VERSION}.csv"
    return os.path.join(CACHE_FOLDER, filename)

def blank_record():
    return {
        "Paper Title": None,
        "Paper Link": None,
        "Publication Year": None,
        "Publication Type": None,
        "Publication Title": None,
        "Author Names": None,
        "DOI": None,
        "PDF Link": None,
        "Open Access": None,
        "Citations Count": 0,
        "PubMed ID": None,
        "PMC ID": None,
        "References": None,
        "arXiv ID": None,
        "OpenAlex ID": None,
        "Source": None,
        "Abstract": None,
        "Review": "NO",
        "Preprint": "NO",
        "arXiv_used": "NO",
    }

# =========================================================
# OPENALEX HELPERS
# =========================================================
def invert_openalex_abstract(abstract_inverted_index):
    """
    Reconstruct abstract text from OpenAlex abstract_inverted_index.
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return None

    try:
        pos_to_word = {}
        max_pos = -1

        for word, positions in abstract_inverted_index.items():
            if not isinstance(positions, list):
                continue
            for pos in positions:
                if isinstance(pos, int):
                    pos_to_word[pos] = word
                    if pos > max_pos:
                        max_pos = pos

        if max_pos < 0:
            return None

        words = [pos_to_word.get(i, "") for i in range(max_pos + 1)]
        text = " ".join(w for w in words if w).strip()
        return re.sub(r"\s+", " ", text) if text else None
    except Exception:
        return None

def extract_openalex_ids(item):
    """
    Extract external IDs from OpenAlex work object.
    """
    ids = item.get("ids") or {}

    pmid = ids.get("pmid")
    pmcid = ids.get("pmcid")
    openalex_id = ids.get("openalex")

    arxiv_id = None

    primary_location = item.get("primary_location") or {}
    landing = primary_location.get("landing_page_url") or ""

    if isinstance(landing, str) and "arxiv.org" in landing:
        arxiv_id = landing.rstrip("/").split("/")[-1]

    if not arxiv_id:
        best_oa_location = item.get("best_oa_location") or {}
        oa_landing = best_oa_location.get("landing_page_url") or ""
        if isinstance(oa_landing, str) and "arxiv.org" in oa_landing:
            arxiv_id = oa_landing.rstrip("/").split("/")[-1]

    return {
        "PubMed ID": pmid,
        "PMC ID": pmcid,
        "arXiv ID": arxiv_id,
        "OpenAlex ID": openalex_id,
    }

def choose_best_paper_link(item):
    """
    Prefer the most useful paper landing page.
    Priority:
    1. DOI URL
    2. primary_location.landing_page_url
    3. best_oa_location.landing_page_url
    4. OpenAlex ID
    """
    doi_url = normalize_doi_to_url(item.get("doi"))
    if doi_url:
        return doi_url

    primary_location = item.get("primary_location") or {}
    best_oa_location = item.get("best_oa_location") or {}

    for candidate in [
        primary_location.get("landing_page_url"),
        best_oa_location.get("landing_page_url"),
        item.get("id"),
    ]:
        if candidate:
            return candidate

    return None

# =========================================================
# QUERY BUILDER
# =========================================================
def build_query_legs(main_keyword, alt_keywords):
    """
    Faster query strategy:
    - each primary alone
    - primary + each alt
    """
    queries = []
    primaries = [k.strip() for k in main_keyword.split(",") if k.strip()]
    alt_keywords = [k.strip() for k in (alt_keywords or []) if k.strip()]

    for primary in primaries:
        queries.append(f'"{primary}"')

        for alt in alt_keywords:
            queries.append(f'"{primary}" AND "{alt}"')

    return list(dict.fromkeys(queries))

def format_query_for_api(query, source):
    if source == "openalex":
        return query.replace('"', "")
    if source == "arxiv":
        return query
    return query

# =========================================================
# SEMANTIC SCHOLAR
# =========================================================
def fetch_semantic_page(session, query, offset):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    try:
        response = session.get(
            url,
            params={
                "query": query,
                "offset": offset,
                "limit": SEMANTIC_PAGE_SIZE,
                "fields": "title,abstract,year,citationCount,externalIds,url,openAccessPdf,authors,venue,isOpenAccess,referenceCount",
            },
            timeout=(5, 15),
        )
        if response.status_code != 200:
            return []
        return response.json().get("data", [])
    except Exception as e:
        print(f"Semantic Scholar page error at offset {offset}: {e}")
        return []

def search_semantic_scholar(query, min_year, max_year):
    session = get_retry_session()
    results = []

    offsets = list(range(0, SEMANTIC_MAX_RESULTS, SEMANTIC_PAGE_SIZE))

    with ThreadPoolExecutor(max_workers=PAGE_THREADS) as executor:
        futures = [executor.submit(fetch_semantic_page, session, query, offset) for offset in offsets]

        for future in as_completed(futures):
            data = future.result()
            for item in data:
                year = item.get("year")
                if not year_is_valid(year, min_year, max_year):
                    continue

                title = item.get("title")
                ext = item.get("externalIds") or {}
                authors = ", ".join(
                    a.get("name") for a in (item.get("authors") or []) if a.get("name")
                ) or None

                record = blank_record()
                record.update({
                    "Paper Title": title,
                    "Paper Link": item.get("url"),
                    "Publication Year": year,
                    "Publication Type": None,
                    "Publication Title": item.get("venue"),
                    "Author Names": authors,
                    "DOI": normalize_doi_to_url(ext.get("DOI")),
                    "PDF Link": (item.get("openAccessPdf") or {}).get("url"),
                    "Open Access": item.get("isOpenAccess"),
                    "Citations Count": safe_int(item.get("citationCount"), 0),
                    "PubMed ID": ext.get("PubMed"),
                    "PMC ID": ext.get("PubMedCentral"),
                    "References": item.get("referenceCount"),
                    "arXiv ID": ext.get("ArXiv"),
                    "OpenAlex ID": None,
                    "Source": "SemanticScholar",
                    "Abstract": item.get("abstract"),
                    "Review": is_review_paper(title),
                    "Preprint": "NO",
                    "arXiv_used": "NO",
                })
                results.append(record)

    return results

# =========================================================
# OPENALEX
# =========================================================
def search_openalex(query, min_year, max_year):
    url = "https://api.openalex.org/works"
    session = get_retry_session()
    results = []
    cursor = "*"

    while len(results) < OPENALEX_MAX_RESULTS:
        try:
            response = session.get(
                url,
                params={
                    "search": query.replace('"', ""),
                    "per-page": 200,
                    "cursor": cursor,
                },
                timeout=(5, 15),
            )

            if response.status_code != 200:
                break

            data = response.json()

            for item in data.get("results", []):
                year = item.get("publication_year")
                if not year_is_valid(year, min_year, max_year):
                    continue

                title = item.get("title") or item.get("display_name")
                review = is_review_paper(title)

                authors = ", ".join(
                    a.get("author", {}).get("display_name")
                    for a in (item.get("authorships") or [])
                    if a.get("author", {}).get("display_name")
                ) or None

                source = (item.get("primary_location") or {}).get("source") or {}
                primary_location = item.get("primary_location") or {}
                best_oa_location = item.get("best_oa_location") or {}

                pdf_link = (
                    primary_location.get("pdf_url")
                    or best_oa_location.get("pdf_url")
                )

                extracted_ids = extract_openalex_ids(item)
                abstract_text = invert_openalex_abstract(item.get("abstract_inverted_index"))

                record = blank_record()
                record.update({
                    "Paper Title": title,
                    "Paper Link": choose_best_paper_link(item),
                    "Publication Year": year,
                    "Publication Type": item.get("type"),
                    "Publication Title": source.get("display_name"),
                    "Author Names": authors,
                    "DOI": normalize_doi_to_url(item.get("doi")),
                    "PDF Link": pdf_link,
                    "Open Access": (item.get("open_access") or {}).get("is_oa"),
                    "Citations Count": safe_int(item.get("cited_by_count"), 0),
                    "PubMed ID": extracted_ids.get("PubMed ID"),
                    "PMC ID": extracted_ids.get("PMC ID"),
                    "References": item.get("referenced_works_count"),
                    "arXiv ID": extracted_ids.get("arXiv ID"),
                    "OpenAlex ID": extracted_ids.get("OpenAlex ID"),
                    "Source": "OpenAlex",
                    "Abstract": abstract_text,
                    "Review": review,
                    "Preprint": "YES" if item.get("type") == "preprint" else "NO",
                    "arXiv_used": "YES" if extracted_ids.get("arXiv ID") else "NO",
                })
                results.append(record)

                if len(results) >= OPENALEX_MAX_RESULTS:
                    break

            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor or len(results) >= OPENALEX_MAX_RESULTS:
                break

        except Exception as e:
            print(f"OpenAlex error: {e}")
            break

    return results

# =========================================================
# arXiv
# =========================================================
def fetch_arxiv_page(session, query, start):
    base_url = "https://export.arxiv.org/api/query"
    try:
        response = session.get(
            base_url,
            params={
                "search_query": f"all:{query}",
                "start": start,
                "max_results": 50,
            },
            timeout=(5, 12),
        )
        if response.status_code != 200:
            return []

        entries = re.findall(r"<entry>(.*?)</entry>", response.text, re.DOTALL)
        return entries
    except Exception as e:
        print(f"arXiv page error at start {start}: {e}")
        return []

def search_arxiv(query, min_year, max_year):
    session = get_retry_session()
    results = []
    starts = list(range(0, ARXIV_MAX_RESULTS, 50))

    with ThreadPoolExecutor(max_workers=PAGE_THREADS) as executor:
        futures = [executor.submit(fetch_arxiv_page, session, query, start) for start in starts]

        for future in as_completed(futures):
            entries = future.result()

            for e in entries:
                title_match = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
                summary_match = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
                published_match = re.search(r"<published>(\d{4})-", e)
                id_match = re.search(r"<id>(.*?)</id>", e)

                year = int(published_match.group(1)) if published_match else None
                if not year_is_valid(year, min_year, max_year):
                    continue

                url = id_match.group(1).strip() if id_match else None
                title = title_match.group(1).strip() if title_match else None
                abstract = re.sub(r"\s+", " ", summary_match.group(1)).strip() if summary_match else None

                authors = re.findall(r"<name>(.*?)</name>", e, re.DOTALL)
                author_names = ", ".join(a.strip() for a in authors) if authors else None

                record = blank_record()
                record.update({
                    "Paper Title": title,
                    "Paper Link": url,
                    "Publication Year": year,
                    "Publication Type": "preprint",
                    "Publication Title": "arXiv",
                    "Author Names": author_names,
                    "DOI": None,
                    "PDF Link": url.replace("/abs/", "/pdf/") if url else None,
                    "Open Access": True,
                    "Citations Count": 0,
                    "PubMed ID": None,
                    "PMC ID": None,
                    "References": None,
                    "arXiv ID": url.split("/")[-1] if url else None,
                    "OpenAlex ID": None,
                    "Source": "arXiv",
                    "Abstract": abstract,
                    "Review": "NO",
                    "Preprint": "YES",
                    "arXiv_used": "YES",
                })
                results.append(record)

    return results

# =========================================================
# RELEVANCE + SCORING
# =========================================================
def normalize_text_for_match(text):
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def keyword_in_text(keyword, text):
    """
    Exact phrase match using word boundaries.
    Prevents false matches like 'amine' in 'examine'
    """
    if not keyword or not text:
        return False

    keyword = normalize_text_for_match(keyword)
    text = normalize_text_for_match(text)

    pattern = r'(?<!\w)' + re.escape(keyword) + r'(?!\w)'
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

def extract_matched_keywords(paper, main_keywords, alt_keywords):
    title = normalize_text_for_match(paper.get("Paper Title"))
    abstract = normalize_text_for_match(paper.get("Abstract"))
    text = f"{title} {abstract}".strip()

    matched_main = []
    matched_alt = []

    for kw in main_keywords:
        kw_clean = kw.strip()
        if kw_clean and keyword_in_text(kw_clean, text):
            matched_main.append(kw_clean)

    for kw in alt_keywords or []:
        kw_clean = kw.strip()
        if kw_clean and keyword_in_text(kw_clean, text):
            matched_alt.append(kw_clean)

    matched_main = list(dict.fromkeys(matched_main))
    matched_alt = list(dict.fromkeys(matched_alt))

    return matched_main, matched_alt

def relevance_score_and_matches(paper, main_keywords, alt_keywords):
    matched_main, matched_alt = extract_matched_keywords(
        paper, main_keywords, alt_keywords
    )

    meets_minimum = len(matched_main) > 0 and len(matched_alt) > 0
    score = (len(matched_main) * 3) + (len(matched_alt) * 1)
    combined_keywords = list(dict.fromkeys(matched_main + matched_alt))

    return {
        "Relevance Score": score,
        "Matched Main Keywords": ", ".join(matched_main) if matched_main else None,
        "Matched Alt Keywords": ", ".join(matched_alt) if matched_alt else None,
        "Matched Keywords Combined": ", ".join(combined_keywords) if combined_keywords else None,
        "Meets Minimum Criteria": "YES" if meets_minimum else "NO",
    }

# =========================================================
# DEDUPLICATION / MERGE
# =========================================================
def merge_records(records):
    merged = {}

    for r in records:
        key = r.get("DOI") or normalize_title(r.get("Paper Title"))
        if not key:
            continue

        if key not in merged:
            merged[key] = dict(r)
            continue

        existing = merged[key]

        existing["Citations Count"] = max(
            safe_int(existing.get("Citations Count"), 0),
            safe_int(r.get("Citations Count"), 0)
        )

        for col in existing.keys():
            if not existing.get(col) and r.get(col):
                existing[col] = r[col]

        if r.get("Preprint") == "YES":
            existing["Preprint"] = "YES"
            existing["arXiv_used"] = "YES"

    return list(merged.values())

# =========================================================
# MAIN
# =========================================================
def run_literature_search(main_keyword, alt_keywords, min_year, max_year):
    os.makedirs(CACHE_FOLDER, exist_ok=True)

    query_legs = build_query_legs(main_keyword, alt_keywords)
    all_results = []

    for query in query_legs:
        cache_path = get_cache_path(query, min_year, max_year)

        if os.path.exists(cache_path):
            print(f"📦 Loading cache: {query}")
            try:
                cached_df = pd.read_csv(cache_path)
                all_results.extend(cached_df.to_dict("records"))
                continue
            except Exception as e:
                print(f"Cache read failed, re-running query. Error: {e}")

        print(f"🔍 Running query: {query}")

        ss_query = format_query_for_api(query, "semantic")
        oa_query = format_query_for_api(query, "openalex")
        ax_query = format_query_for_api(query, "arxiv")

        with ThreadPoolExecutor(max_workers=API_THREADS) as executor:
            ss_future = executor.submit(search_semantic_scholar, ss_query, min_year, max_year)
            oa_future = executor.submit(search_openalex, oa_query, min_year, max_year)
            ax_future = executor.submit(search_arxiv, ax_query, min_year, max_year)

            combined = ss_future.result() + oa_future.result() + ax_future.result()

        if combined:
            try:
                pd.DataFrame(combined).to_csv(cache_path, index=False)
            except Exception as e:
                print(f"Cache write failed for query {query}: {e}")

        all_results.extend(combined)

    merged = merge_records(all_results)
    df = pd.DataFrame(merged)

    if df.empty:
        return df

    main_list = [k.strip() for k in main_keyword.split(",") if k.strip()]
    alt_list = [k.strip() for k in (alt_keywords or []) if k.strip()]

    df["Citations Count"] = pd.to_numeric(df["Citations Count"], errors="coerce").fillna(0)

    relevance_results = df.apply(
        lambda row: relevance_score_and_matches(row, main_list, alt_list),
        axis=1
    )

    relevance_df = pd.DataFrame(list(relevance_results))

    df["Relevance Score"] = relevance_df["Relevance Score"]
    df["Matched Main Keywords"] = relevance_df["Matched Main Keywords"]
    df["Matched Alt Keywords"] = relevance_df["Matched Alt Keywords"]
    df["Matched Keywords Combined"] = relevance_df["Matched Keywords Combined"]
    df["Meets Minimum Criteria"] = relevance_df["Meets Minimum Criteria"]

    # Remove rows with Relevance Score = 0
    df = df[df["Relevance Score"] > 0].copy()

    # Sort by Citation Count first, then Relevance Score
    df = df.sort_values(
        by=["Citations Count", "Relevance Score"],
        ascending=False
    ).reset_index(drop=True)

    # Remove unwanted output columns
    df = df.drop(
        columns=[
            "Matched Main Keywords",
            "Matched Alt Keywords",
        ],
        errors="ignore"
    )

    return df

