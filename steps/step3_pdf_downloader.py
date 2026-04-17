import os
import re
import requests
import streamlit as st
import pandas as pd

from time import sleep
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


# ================= CONFIG =================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

UNPAYWALL_EMAIL = "ysalvekar@ingenero.com"
REQUEST_TIMEOUT = 30
DEFAULT_MAX_WORKERS = 5


# ================= HELPERS =================
def safe_filename(text):
    text = str(text) if text is not None else "paper"
    cleaned = "".join(c for c in text if c.isalnum() or c in (" ", "_", "-")).rstrip()
    return cleaned or "paper"


def clean_value(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return value


def is_valid_url(url):
    url = clean_value(url)
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


def normalize_doi_url(doi):
    doi = clean_value(doi)
    if not doi:
        return None

    doi = (
        doi.replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("doi:", "")
        .strip()
    )
    if not doi:
        return None

    return f"https://doi.org/{doi}"


def is_probably_pdf(resp):
    ctype = resp.headers.get("Content-Type", "").lower()
    final_url = str(resp.url).lower()
    disposition = resp.headers.get("Content-Disposition", "").lower()

    return (
        "application/pdf" in ctype
        or final_url.endswith(".pdf")
        or ".pdf?" in final_url
        or "filename=" in disposition and ".pdf" in disposition
    )


def classify_failure_reason(error_text):
    text = str(error_text).lower()

    if "403" in text or "forbidden" in text:
        return "FORBIDDEN_403"
    if "404" in text:
        return "NOT_FOUND_404"
    if "timeout" in text:
        return "TIMEOUT"
    if "ssl" in text:
        return "SSL_ERROR"
    if "invalid url" in text:
        return "INVALID_URL"
    if "not_pdf_response" in text:
        return "NOT_PDF_RESPONSE"
    if "html_no_pdf" in text:
        return "HTML_NO_PDF"
    if "article_page_no_pdf" in text:
        return "ARTICLE_PAGE_NO_PDF"
    if "sciencedirect_no_pii" in text:
        return "SCIENCEDIRECT_NO_PII"
    if "sciencedirect_failed" in text:
        return "SCIENCEDIRECT_FAILED"
    if "sciencedirect_resolve_failed" in text:
        return "SCIENCEDIRECT_RESOLVE_FAILED"
    if "no_open_access_pdf" in text or "no open access pdf" in text:
        return "NO_OPEN_ACCESS_PDF"
    if "unpaywall" in text:
        return "UNPAYWALL_ERROR"
    if "sciencedirect_public_route_failed" in text:
        return "SCIENCEDIRECT_PUBLIC_ROUTE_FAILED"
    if "all sources failed" in text:
        return "ALL_SOURCES_FAILED"

    return "OTHER_DOWNLOAD_ERROR"


def get_requests_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def set_last_error(current_error, new_error):
    return new_error if new_error else current_error


def build_attempted_sources(candidate_attempts):
    if not candidate_attempts:
        return None
    return " | ".join([f"{src}: {url}" for src, url in candidate_attempts])


def success_response(record, path, title, source, mode, final_url, candidate_attempts, message):
    record.update({
        "download_status": "success",
        "resolved_pdf_url": final_url,
        "saved_pdf_path": path,
        "failure_reason": None,
        "failure_category": None,
        "download_source": source,
        "download_mode": mode,
        "manual_download_recommended": "NO",
        "attempted_sources": build_attempted_sources(candidate_attempts),
    })
    return {
        "status": "success",
        "record": record,
        "saved_pdf_path": path,
        "message": message,
    }


# ================= DOWNLOAD / EXTRACTION =================
def try_direct_download(url, path, session=None, referer=None):
    if not is_valid_url(url):
        return None, "INVALID_URL", None

    session = session or get_requests_session()

    headers = HEADERS.copy()
    if referer:
        headers["Referer"] = referer
        parsed = urlparse(referer)
        headers["Origin"] = f"{parsed.scheme}://{parsed.netloc}"

    r = session.get(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        stream=True,
        allow_redirects=True,
    )
    r.raise_for_status()

    if not is_probably_pdf(r):
        return None, "NOT_PDF_RESPONSE", r.url

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return path, "DIRECT", r.url


def extract_pdf_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")

    # Standard metadata
    meta_names = [
        "citation_pdf_url",
        "wkhealth_pdf_url",
        "pdf_url",
    ]
    for name in meta_names:
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            return urljoin(base_url, meta["content"])

    # Look for obvious href/src/data links
    for tag in soup.find_all(["a", "iframe", "embed", "object"]):
        href = tag.get("href") or tag.get("src") or tag.get("data")
        if href:
            href_low = href.lower()
            if ".pdf" in href_low or "/pdf" in href_low or "/pdfft" in href_low:
                return urljoin(base_url, href)

    # Script/JSON patterns
    patterns = [
        r'"pdfUrl":"([^"]+)"',
        r'"downloadPdfUrl":"([^"]+)"',
        r'"linkToPdf":"([^"]+)"',
        r'"pdfDownload":\{"linkToPdf":"([^"]+)"',
        r'"url":"([^"]*\/pdfft\?[^"]+)"',
        r'"url":"([^"]*\/pdf[^"]*)"',
        r'https?://[^\s"\']+\.pdf(?:\?[^\s"\']*)?',
        r'https?://[^\s"\']+/pdfft\?[^\s"\']+',
    ]

    for pattern in patterns:
        m = re.search(pattern, html, flags=re.I)
        if m:
            found = m.group(1) if m.groups() else m.group(0)
            found = found.replace("\\u002F", "/").replace("\\/", "/")
            return urljoin(base_url, found)

    return None


def try_html_fallback(url, session=None):
    if not is_valid_url(url):
        return None, "INVALID_URL"

    session = session or get_requests_session()

    r = session.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    r.raise_for_status()

    pdf_url = extract_pdf_from_html(r.text, r.url)
    if not pdf_url:
        return None, "HTML_NO_PDF"

    return pdf_url, "HTML_EXTRACTED"


def try_article_page_pdf_extraction(source_url, path, session=None):
    if not is_valid_url(source_url):
        return None, "INVALID_URL", None

    session = session or get_requests_session()

    r = session.get(
        source_url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    r.raise_for_status()

    article_url = r.url
    pdf_url = extract_pdf_from_html(r.text, article_url)
    if not pdf_url:
        return None, "ARTICLE_PAGE_NO_PDF", article_url

    downloaded_path, mode, final_pdf_url = try_direct_download(
        pdf_url,
        path,
        session=session,
        referer=article_url,
    )

    if downloaded_path:
        return downloaded_path, "ARTICLE_PAGE_EXTRACTED", final_pdf_url

    return None, mode, final_pdf_url


def resolve_final_url(url, session=None):
    if not is_valid_url(url):
        return None

    session = session or get_requests_session()

    try:
        r = session.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=False,
        )
        r.raise_for_status()
        return r.url
    except Exception:
        return None


def extract_sciencedirect_pii(url):
    url = clean_value(url)
    if not url:
        return None

    m = re.search(r"/pii/([A-Z0-9]+)", url, flags=re.I)
    if m:
        return m.group(1)

    return None


def build_sciencedirect_pdf_candidates(pii):
    pii = clean_value(pii)
    if not pii:
        return []

    return [
        f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true",
        f"https://www.sciencedirect.com/science/article/pii/{pii}/pdf?md5=&pid=1-s2.0-{pii}-main.pdf",
    ]


def try_sciencedirect_fallback(source_url, path, session=None):
    if not is_valid_url(source_url):
        return None, "INVALID_URL", None

    session = session or get_requests_session()

    final_url = resolve_final_url(source_url, session=session)
    if not final_url:
        return None, "SCIENCEDIRECT_RESOLVE_FAILED", None

    pii = extract_sciencedirect_pii(final_url)
    if not pii:
        return None, "SCIENCEDIRECT_NO_PII", final_url

    candidates = build_sciencedirect_pdf_candidates(pii)

    last_err = None
    for candidate in candidates:
        try:
            downloaded_path, mode, final_pdf_url = try_direct_download(
                candidate,
                path,
                session=session,
                referer=final_url,
            )
            if downloaded_path:
                return downloaded_path, "SCIENCEDIRECT_PII", final_pdf_url
            last_err = set_last_error(last_err, mode)
        except Exception as e:
            last_err = e

    return None, f"SCIENCEDIRECT_PUBLIC_ROUTE_FAILED: {last_err}", final_url


def get_unpaywall_pdf(doi, session=None):
    doi_url = normalize_doi_url(doi)
    if not doi_url:
        return None, "NO_VALID_DOI"

    doi_value = doi_url.replace("https://doi.org/", "")
    session = session or get_requests_session()

    try:
        url = f"https://api.unpaywall.org/v2/{doi_value}?email={UNPAYWALL_EMAIL}"
        r = session.get(url, timeout=20)

        if r.status_code != 200:
            return None, f"UNPAYWALL_API_{r.status_code}"

        data = r.json()

        # best_oa_location.url_for_pdf
        best = data.get("best_oa_location") or {}
        pdf = clean_value(best.get("url_for_pdf"))
        if pdf:
            return pdf, None

        # best_oa_location.url
        best_url = clean_value(best.get("url"))
        if best_url:
            return best_url, "UNPAYWALL_BEST_URL_NO_DIRECT_PDF"

        # oa_locations fallback
        for loc in data.get("oa_locations", []) or []:
            pdf = clean_value(loc.get("url_for_pdf"))
            if pdf:
                return pdf, None

        for loc in data.get("oa_locations", []) or []:
            loc_url = clean_value(loc.get("url"))
            if loc_url:
                return loc_url, "UNPAYWALL_OA_URL_NO_DIRECT_PDF"

        return None, "NO_OPEN_ACCESS_PDF"

    except Exception as e:
        return None, f"UNPAYWALL_FAILED: {str(e)}"


def get_arxiv_pdf(arxiv_id):
    arxiv_id = clean_value(arxiv_id)
    if not arxiv_id:
        return None
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


# ================= SINGLE PAPER WORKER =================
def process_single_paper(index, row_dict, output_dir):
    session = get_requests_session()
    record = dict(row_dict)

    title = clean_value(record.get("Paper Title")) or "paper"
    pdf_link = clean_value(record.get("PDF Link"))
    doi = clean_value(record.get("DOI"))
    paper_link = clean_value(record.get("Paper Link"))
    arxiv_id = clean_value(record.get("arXiv ID"))
    doi_url = normalize_doi_url(doi)

    fname = f"{index}_{safe_filename(title)[:100]}.pdf"
    path = os.path.join(output_dir, fname)

    candidate_attempts = []
    last_error = None

    try:
        # ---------- 1️⃣ Direct PDF Link ----------
        if is_valid_url(pdf_link):
            candidate_attempts.append(("PDF Link", pdf_link))
            try:
                direct_path, mode, final_url = try_direct_download(
                    pdf_link, path, session=session
                )
                if direct_path:
                    return success_response(
                        record, path, title, "PDF Link", mode, final_url,
                        candidate_attempts, f"✅ Downloaded: {title}"
                    )
                last_error = set_last_error(last_error, mode)
            except Exception as e:
                last_error = e

        # ---------- 2️⃣ HTML fallback from PDF Link ----------
        if is_valid_url(pdf_link):
            try:
                pdf_url_html, reason = try_html_fallback(pdf_link, session=session)
                if pdf_url_html:
                    candidate_attempts.append(("HTML fallback from PDF Link", pdf_url_html))
                    direct_path, mode, final_url = try_direct_download(
                        pdf_url_html, path, session=session, referer=pdf_link
                    )
                    if direct_path:
                        return success_response(
                            record, path, title, "HTML fallback from PDF Link", mode, final_url,
                            candidate_attempts, f"✅ Downloaded (HTML fallback): {title}"
                        )
                    last_error = set_last_error(last_error, mode)
                else:
                    last_error = set_last_error(last_error, reason)
            except Exception as e:
                last_error = e

        # ---------- 3️⃣ arXiv fallback ----------
        arxiv_pdf = get_arxiv_pdf(arxiv_id)
        if is_valid_url(arxiv_pdf):
            candidate_attempts.append(("arXiv", arxiv_pdf))
            try:
                direct_path, mode, final_url = try_direct_download(
                    arxiv_pdf, path, session=session
                )
                if direct_path:
                    return success_response(
                        record, path, title, "arXiv", mode, final_url,
                        candidate_attempts, f"✅ Downloaded (arXiv): {title}"
                    )
                last_error = set_last_error(last_error, mode)
            except Exception as e:
                last_error = e

        # ---------- 4️⃣ Unpaywall fallback ----------
        if doi:
            unpaywall_url, reason = get_unpaywall_pdf(doi, session=session)
            record["unpaywall_url"] = unpaywall_url

            if is_valid_url(unpaywall_url):
                candidate_attempts.append(("Unpaywall", unpaywall_url))
                try:
                    direct_path, mode, final_url = try_direct_download(
                        unpaywall_url, path, session=session
                    )
                    if direct_path:
                        return success_response(
                            record, path, title, "Unpaywall", mode, final_url,
                            candidate_attempts, f"✅ Downloaded (Unpaywall): {title}"
                        )
                    last_error = set_last_error(last_error, mode)
                except Exception as e:
                    last_error = e

                # HTML extraction from Unpaywall URL
                try:
                    pdf_url_html, reason2 = try_html_fallback(unpaywall_url, session=session)
                    if pdf_url_html:
                        candidate_attempts.append(("HTML fallback from Unpaywall URL", pdf_url_html))
                        direct_path, mode, final_url = try_direct_download(
                            pdf_url_html, path, session=session, referer=unpaywall_url
                        )
                        if direct_path:
                            return success_response(
                                record, path, title, "HTML fallback from Unpaywall URL", mode, final_url,
                                candidate_attempts, f"✅ Downloaded (Unpaywall HTML fallback): {title}"
                            )
                        last_error = set_last_error(last_error, mode)
                    else:
                        last_error = set_last_error(last_error, reason2)
                except Exception as e:
                    last_error = e
            else:
                last_error = set_last_error(last_error, reason)

        # ---------- 5️⃣ DOI landing page fallback ----------
        if is_valid_url(doi_url):
            try:
                pdf_url_html, reason = try_html_fallback(doi_url, session=session)
                if pdf_url_html:
                    candidate_attempts.append(("DOI landing page", pdf_url_html))
                    direct_path, mode, final_url = try_direct_download(
                        pdf_url_html, path, session=session, referer=doi_url
                    )
                    if direct_path:
                        return success_response(
                            record, path, title, "DOI landing page", mode, final_url,
                            candidate_attempts, f"✅ Downloaded (DOI landing page): {title}"
                        )
                    last_error = set_last_error(last_error, mode)
                else:
                    last_error = set_last_error(last_error, reason)
            except Exception as e:
                last_error = e

        # ---------- 6️⃣ Paper Link landing page fallback ----------
        if is_valid_url(paper_link):
            try:
                pdf_url_html, reason = try_html_fallback(paper_link, session=session)
                if pdf_url_html:
                    candidate_attempts.append(("Paper Link landing page", pdf_url_html))
                    direct_path, mode, final_url = try_direct_download(
                        pdf_url_html, path, session=session, referer=paper_link
                    )
                    if direct_path:
                        return success_response(
                            record, path, title, "Paper Link landing page", mode, final_url,
                            candidate_attempts, f"✅ Downloaded (Paper Link fallback): {title}"
                        )
                    last_error = set_last_error(last_error, mode)
                else:
                    last_error = set_last_error(last_error, reason)
            except Exception as e:
                last_error = e

        # ---------- 7️⃣ Article-page extraction from DOI ----------
        if is_valid_url(doi_url):
            try:
                candidate_attempts.append(("Article page extraction from DOI", doi_url))
                direct_path, mode, final_url = try_article_page_pdf_extraction(
                    doi_url, path, session=session
                )
                if direct_path:
                    return success_response(
                        record, path, title, "Article page extraction from DOI", mode, final_url,
                        candidate_attempts, f"✅ Downloaded (article-page extraction DOI): {title}"
                    )
                last_error = set_last_error(last_error, mode)
            except Exception as e:
                last_error = e

        # ---------- 8️⃣ Article-page extraction from Paper Link ----------
        if is_valid_url(paper_link):
            try:
                candidate_attempts.append(("Article page extraction from Paper Link", paper_link))
                direct_path, mode, final_url = try_article_page_pdf_extraction(
                    paper_link, path, session=session
                )
                if direct_path:
                    return success_response(
                        record, path, title, "Article page extraction from Paper Link", mode, final_url,
                        candidate_attempts, f"✅ Downloaded (article-page extraction Paper Link): {title}"
                    )
                last_error = set_last_error(last_error, mode)
            except Exception as e:
                last_error = e

        # ---------- 9️⃣ ScienceDirect / Elsevier PII fallback ----------
        sciencedirect_sources = []
        if is_valid_url(doi_url):
            sciencedirect_sources.append(("ScienceDirect fallback from DOI", doi_url))
        if is_valid_url(paper_link):
            sciencedirect_sources.append(("ScienceDirect fallback from Paper Link", paper_link))
        if is_valid_url(pdf_link):
            sciencedirect_sources.append(("ScienceDirect fallback from PDF Link", pdf_link))

        for label, src_url in sciencedirect_sources:
            try:
                candidate_attempts.append((label, src_url))
                direct_path, mode, final_url = try_sciencedirect_fallback(
                    src_url, path, session=session
                )
                if direct_path:
                    return success_response(
                        record, path, title, label, mode, final_url,
                        candidate_attempts, f"✅ Downloaded ({label}): {title}"
                    )
                last_error = set_last_error(last_error, mode)
            except Exception as e:
                last_error = e

        raise Exception(last_error if last_error else "All sources failed")

    except Exception as e:
        record.update({
            "download_status": "failed",
            "resolved_pdf_url": None,
            "saved_pdf_path": None,
            "failure_reason": str(e),
            "failure_category": classify_failure_reason(str(e)),
            "download_source": "None",
            "download_mode": None,
            "manual_download_recommended": "YES",
            "attempted_sources": build_attempted_sources(candidate_attempts),
        })
        return {
            "status": "failed",
            "record": record,
            "saved_pdf_path": None,
            "message": f"❌ Failed: {title} — {e}",
        }


# ================= MAIN FUNCTION =================
def download_pdfs(
    df,
    output_dir="outputs/pdfs",
    report_path="outputs/pdf_download_report.xlsx",
    delay=0.0,
    max_workers=DEFAULT_MAX_WORKERS,
):
    os.makedirs(output_dir, exist_ok=True)

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    st.subheader("📥 Step 3 — Download PDFs")

    results = []
    downloaded_paths = []

    total = len(df)
    progress = st.progress(0)
    status_box = st.empty()

    if total == 0:
        report_df = pd.DataFrame()
        report_df.to_excel(report_path, index=False)
        st.info(f"📄 Download report saved to: {report_path}")
        return downloaded_paths, report_df

    max_workers = max(1, min(int(max_workers), total))
    rows = [(i, row.to_dict()) for i, (_, row) in enumerate(df.iterrows(), start=1)]

    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_paper, i, row_dict, output_dir): i
            for i, row_dict in rows
        }

        for future in as_completed(futures):
            result = future.result()

            results.append(result["record"])

            if result["saved_pdf_path"]:
                downloaded_paths.append(result["saved_pdf_path"])

            completed += 1
            progress.progress(completed / total)
            status_box.info(
                f"Processed {completed}/{total} papers | "
                f"Downloaded: {len(downloaded_paths)} | "
                f"Workers: {max_workers}"
            )

            if result["status"] == "success":
                st.success(result["message"])
            else:
                st.error(result["message"])

            if delay > 0:
                sleep(delay)

    report_df = pd.DataFrame(results)

    if not report_df.empty:
        report_df.to_excel(report_path, index=False)

    st.info(f"📄 Download report saved to: {report_path}")

    return downloaded_paths, report_df
