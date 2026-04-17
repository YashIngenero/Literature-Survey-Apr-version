import os
import re
import requests
import streamlit as st
import pandas as pd

from time import sleep
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


# ================= CONFIG =================
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
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

    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "").strip()
    if not doi:
        return None

    return f"https://doi.org/{doi}"


def is_probably_pdf(resp):
    ctype = resp.headers.get("Content-Type", "").lower()
    final_url = str(resp.url).lower()
    return "application/pdf" in ctype or final_url.endswith(".pdf")


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
    if "no_open_access_pdf" in text or "no open access pdf" in text:
        return "NO_OPEN_ACCESS_PDF"
    if "unpaywall" in text:
        return "UNPAYWALL_ERROR"
    if "all sources failed" in text:
        return "ALL_SOURCES_FAILED"

    return "OTHER_DOWNLOAD_ERROR"


def get_requests_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def try_direct_download(url, path, session=None, referer=None):
    if not is_valid_url(url):
        return None, "INVALID_URL", None

    session = session or get_requests_session()

    headers = HEADERS.copy()
    if referer:
        headers["Referer"] = referer

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

    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content"):
        return urljoin(base_url, meta["content"])

    for tag in soup.find_all(["a", "iframe", "embed", "object"]):
        href = tag.get("href") or tag.get("src") or tag.get("data")
        if href and ".pdf" in href.lower():
            return urljoin(base_url, href)

    matches = re.findall(r"https?://[^\s\"']+\.pdf", html)
    if matches:
        return matches[0]

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

        if data.get("best_oa_location"):
            pdf = clean_value(data["best_oa_location"].get("url_for_pdf"))
            if pdf:
                return pdf, None

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
                    record.update({
                        "download_status": "success",
                        "resolved_pdf_url": final_url,
                        "saved_pdf_path": path,
                        "failure_reason": None,
                        "failure_category": None,
                        "download_source": "PDF Link",
                        "download_mode": mode,
                        "manual_download_recommended": "NO",
                        "attempted_sources": " | ".join(
                            [f"{src}: {url}" for src, url in candidate_attempts]
                        ),
                    })
                    return {
                        "status": "success",
                        "record": record,
                        "saved_pdf_path": path,
                        "message": f"✅ Downloaded: {title}",
                    }
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
                        record.update({
                            "download_status": "success",
                            "resolved_pdf_url": final_url,
                            "saved_pdf_path": path,
                            "failure_reason": None,
                            "failure_category": None,
                            "download_source": "HTML fallback from PDF Link",
                            "download_mode": mode,
                            "manual_download_recommended": "NO",
                            "attempted_sources": " | ".join(
                                [f"{src}: {url}" for src, url in candidate_attempts]
                            ),
                        })
                        return {
                            "status": "success",
                            "record": record,
                            "saved_pdf_path": path,
                            "message": f"✅ Downloaded (HTML fallback): {title}",
                        }
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
                    record.update({
                        "download_status": "success",
                        "resolved_pdf_url": final_url,
                        "saved_pdf_path": path,
                        "failure_reason": None,
                        "failure_category": None,
                        "download_source": "arXiv",
                        "download_mode": mode,
                        "manual_download_recommended": "NO",
                        "attempted_sources": " | ".join(
                            [f"{src}: {url}" for src, url in candidate_attempts]
                        ),
                    })
                    return {
                        "status": "success",
                        "record": record,
                        "saved_pdf_path": path,
                        "message": f"✅ Downloaded (arXiv): {title}",
                    }
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
                        record.update({
                            "download_status": "success",
                            "resolved_pdf_url": final_url,
                            "saved_pdf_path": path,
                            "failure_reason": None,
                            "failure_category": None,
                            "download_source": "Unpaywall",
                            "download_mode": mode,
                            "manual_download_recommended": "NO",
                            "attempted_sources": " | ".join(
                                [f"{src}: {url}" for src, url in candidate_attempts]
                            ),
                        })
                        return {
                            "status": "success",
                            "record": record,
                            "saved_pdf_path": path,
                            "message": f"✅ Downloaded (Unpaywall): {title}",
                        }
                except Exception as e:
                    last_error = e
            else:
                if reason:
                    last_error = reason

        # ---------- 5️⃣ DOI landing page fallback ----------
        doi_url = normalize_doi_url(doi)
        if is_valid_url(doi_url):
            try:
                pdf_url_html, reason = try_html_fallback(doi_url, session=session)
                if pdf_url_html:
                    candidate_attempts.append(("DOI landing page", pdf_url_html))
                    direct_path, mode, final_url = try_direct_download(
                        pdf_url_html, path, session=session, referer=doi_url
                    )
                    if direct_path:
                        record.update({
                            "download_status": "success",
                            "resolved_pdf_url": final_url,
                            "saved_pdf_path": path,
                            "failure_reason": None,
                            "failure_category": None,
                            "download_source": "DOI landing page",
                            "download_mode": mode,
                            "manual_download_recommended": "NO",
                            "attempted_sources": " | ".join(
                                [f"{src}: {url}" for src, url in candidate_attempts]
                            ),
                        })
                        return {
                            "status": "success",
                            "record": record,
                            "saved_pdf_path": path,
                            "message": f"✅ Downloaded (DOI landing page): {title}",
                        }
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
                        record.update({
                            "download_status": "success",
                            "resolved_pdf_url": final_url,
                            "saved_pdf_path": path,
                            "failure_reason": None,
                            "failure_category": None,
                            "download_source": "Paper Link landing page",
                            "download_mode": mode,
                            "manual_download_recommended": "NO",
                            "attempted_sources": " | ".join(
                                [f"{src}: {url}" for src, url in candidate_attempts]
                            ),
                        })
                        return {
                            "status": "success",
                            "record": record,
                            "saved_pdf_path": path,
                            "message": f"✅ Downloaded (Paper Link fallback): {title}",
                        }
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
            "attempted_sources": " | ".join(
                [f"{src}: {url}" for src, url in candidate_attempts]
            ) if candidate_attempts else None,
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
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

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

    # Sort report by original processing index if available from saved path prefix
    def extract_index(rec):
        path = str(rec.get("saved_pdf_path") or "")
        if path:
            name = os.path.basename(path)
            m = re.match(r"^(\d+)_", name)
            if m:
                return int(m.group(1))
        return 10**9

    report_df = pd.DataFrame(results)

    if not report_df.empty:
        # keep a more stable readable order if original title rows got mixed by parallel completion
        report_df["_sort_order"] = range(len(report_df))
        report_df.to_excel(report_path, index=False)

        if "_sort_order" in report_df.columns:
            report_df = report_df.drop(columns=["_sort_order"], errors="ignore")
            report_df.to_excel(report_path, index=False)

    st.info(f"📄 Download report saved to: {report_path}")

    return downloaded_paths, report_df