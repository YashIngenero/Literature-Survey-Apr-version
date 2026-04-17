import os
import io
import re
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st
from spellchecker import SpellChecker

from steps.step1_literature_search import run_literature_search
from steps.step2_filter_ui import step2_filter_ui
from steps.step3_pdf_downloader import download_pdfs
from steps.step4_pdf_summarizer import process_papers, DEFAULT_MODEL, FALLBACK_MODEL
from utils.file_utils import create_zip
from utils.io_helpers import ensure_dir


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Literature Survey Automation", layout="wide")
st.title("📚 Literature Survey Automation Platform Version 3")


# =====================================================
# OUTPUT DIRECTORIES
# =====================================================
BASE_OUTPUT_DIR = "outputs"
SEARCH_DIR = ensure_dir(os.path.join(BASE_OUTPUT_DIR, "search_results"))
FILTER_DIR = ensure_dir(os.path.join(BASE_OUTPUT_DIR, "filtered_results"))
PDF_DIR = ensure_dir(os.path.join(BASE_OUTPUT_DIR, "pdfs"))
SUMMARY_DIR = ensure_dir(os.path.join(BASE_OUTPUT_DIR, "summaries"))


# =====================================================
# HELPERS
# =====================================================
def apply_correction(corrected_text: str):
    st.session_state.query_input = corrected_text


def normalize_selected_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Selected" in df.columns:
        df["Selected"] = (
            df["Selected"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": True,
                "true": True,
                "1": True,
                "y": True,
                "no": False,
                "false": False,
                "0": False,
                "n": False,
            })
        )
        df["Selected"] = df["Selected"].fillna(False)
    else:
        df.insert(0, "Selected", False)

    return df


def safe_query_filename(query_text: str) -> str:
    safe_query = re.sub(r"[^\w\s-]", "", query_text).strip().replace(" ", "_")
    return safe_query if safe_query else "results"


def create_zip_from_dict(files_dict: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname, fbytes in files_dict.items():
            zipf.writestr(fname, fbytes)
    buffer.seek(0)
    return buffer


def load_step3_pdfs_as_inputs(pdf_paths):
    pdf_inputs = []

    for p in pdf_paths:
        try:
            with open(p, "rb") as f:
                file_obj = io.BytesIO(f.read())
                file_obj.name = os.path.basename(p)
                pdf_inputs.append(file_obj)
        except Exception as e:
            st.error(f"Could not load {os.path.basename(p)}: {e}")

    return pdf_inputs


def load_step3_pdfs_for_zip(pdf_paths):
    files_for_zip = {}

    for p in pdf_paths:
        try:
            with open(p, "rb") as f:
                files_for_zip[os.path.basename(p)] = f.read()
        except Exception as e:
            st.error(f"Could not add {os.path.basename(p)} to ZIP: {e}")

    return files_for_zip


def normalize_title_key(title: str) -> str:
    if not title:
        return ""
    return re.sub(r"\W+", "", str(title).lower()).strip()


def get_paper_key(row) -> str:
    doi = str(row.get("DOI", "")).strip().lower()
    if doi and doi != "nan":
        return f"doi::{doi}"

    title_key = normalize_title_key(row.get("Paper Title", ""))
    return f"title::{title_key}" if title_key else ""


def get_step3_all_pdf_paths():
    pdf_map = st.session_state.get("step3_pdf_map", {})
    return [
        item["pdf_path"]
        for item in pdf_map.values()
        if item.get("pdf_path") and os.path.exists(item["pdf_path"])
    ]


def rebuild_downloaded_pdfs_from_map():
    st.session_state["downloaded_pdfs"] = get_step3_all_pdf_paths()


def rebuild_summaries_from_map():
    summary_map = st.session_state.get("step4_summary_map", {})
    st.session_state["summaries"] = {
        item["summary_name"]: item["summary_bytes"]
        for item in summary_map.values()
        if item.get("summary_name") and item.get("summary_bytes") is not None
    }


# =====================================================
# SESSION STATE DEFAULTS
# =====================================================
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if "summaries" not in st.session_state:
    st.session_state["summaries"] = {}

if "step3_pdf_map" not in st.session_state:
    st.session_state["step3_pdf_map"] = {}

if "step4_summary_map" not in st.session_state:
    st.session_state["step4_summary_map"] = {}


# =====================================================
# STEP 1 — SEARCH
# =====================================================
st.header("Step 1 — Literature Search")

query = st.text_input(
    "Enter primary keyword(s) (comma-separated)",
    key="query_input",
    placeholder="e.g. Carbon Capture, CCU",
)

alt_keywords_input = st.text_input(
    "Additional alternate keywords (comma-separated, optional)",
    placeholder="e.g. Primary amine, Secondary amine, blends, CO2 loading",
)

# ---------------- SPELL CHECK ----------------
spell = SpellChecker()

if query.strip():
    raw_primary_keywords = [k.strip() for k in query.split(",") if k.strip()]
    corrected_primary_keywords = []

    for kw in raw_primary_keywords:
        words = kw.split()
        misspelled = spell.unknown(words)

        corrected_words = []
        for word in words:
            if word in misspelled:
                suggestion = spell.correction(word)
                corrected_words.append(suggestion if suggestion else word)
            else:
                corrected_words.append(word)

        corrected_primary_keywords.append(" ".join(corrected_words))

    corrected_query = ", ".join(corrected_primary_keywords)

    if corrected_query != query:
        st.warning(f"Did you mean: **{corrected_query}** ?")
        st.button(
            "Apply Correction",
            on_click=apply_correction,
            args=(corrected_query,),
            key="apply_query_correction",
        )

# ---------------- YEAR FILTERS ----------------
current_year = datetime.now().year
year_options = list(range(current_year, 1990, -1))

col1, col2 = st.columns(2)

with col1:
    min_year = st.selectbox(
        "Minimum publication year",
        options=year_options,
        index=year_options.index(2016) if 2016 in year_options else 0,
        key="step1_min_year",
    )

with col2:
    max_year = st.selectbox(
        "Maximum publication year",
        options=year_options,
        index=0,
        key="step1_max_year",
    )

# ---------------- PROCESS KEYWORDS ----------------
primary_keywords = [k.strip() for k in query.split(",") if k.strip()]
alt_keywords = [k.strip() for k in alt_keywords_input.split(",") if k.strip()]

# ---------------- QUERY LEG PREVIEW ----------------
if primary_keywords:
    query_legs_preview = []

    for primary in primary_keywords:
        query_legs_preview.append(f'"{primary}"')
        for alt in alt_keywords:
            query_legs_preview.append(f'"{primary}" AND "{alt}"')

    st.info(f"🔑 Total query legs to run: {len(query_legs_preview)}")
    st.code("\n".join(query_legs_preview[:20]))

    if len(query_legs_preview) > 20:
        st.caption("Showing first 20 query legs only.")

# ---------------- VALIDATION ----------------
error_message = None

if not primary_keywords:
    error_message = "At least one primary keyword is required."
elif min_year > max_year:
    error_message = "Minimum year cannot be greater than maximum year."

if error_message:
    st.error(error_message)

# ---------------- RUN SEARCH ----------------
search_disabled = error_message is not None

if st.button("🔍 Run Search", disabled=search_disabled, key="run_step1_search"):
    with st.spinner("Searching literature sources..."):
        df = run_literature_search(
            main_keyword=query,
            alt_keywords=alt_keywords,
            min_year=min_year,
            max_year=max_year,
        )

        if isinstance(df, tuple):
            df = df[0]

        if df.empty:
            st.warning("No results found. Try broader keywords.")
            st.stop()

        st.session_state["step1_df"] = df

        path = os.path.join(SEARCH_DIR, "step1_raw_results.xlsx")
        df.to_excel(path, index=False)

# ---------------- DISPLAY RESULTS ----------------
if "step1_df" in st.session_state:
    df = st.session_state["step1_df"]

    st.success(f"{len(df)} papers retrieved.")

    #st.subheader("📊 Source Distribution")
    #st.bar_chart(df["Source"].value_counts())

    #st.subheader("🏆 Top 10 Most Cited Papers")
    #st.dataframe(df.head(10), use_container_width=True)

    #st.subheader("📄 All Results")
    #st.dataframe(df, use_container_width=True)
    with st.expander("📄 All Results", expanded=False):
    st.dataframe(df, use_container_width=True)

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    query_for_filename = st.session_state.get("query_input", "").strip()
    safe_query = safe_query_filename(query_for_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"Raw_{safe_query}_{timestamp}.xlsx"

    st.download_button(
        "⬇ Download Step 1 Results (Excel)",
        data=buffer,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_step1_excel",
    )

st.divider()


# =====================================================
# STEP 2 — FILTER, SELECT, OR UPLOAD
# =====================================================
st.header("Step 2 — Filter & Select Papers")

source_option = st.radio(
    "Source of paper list",
    ["From Step 1", "Upload filtered Excel"],
    horizontal=True,
    key="step2_source_option",
)

full_df = None
selected_df = None

# ---------------- UPLOAD OPTION ----------------
if source_option == "Upload filtered Excel":
    uploaded_file = st.file_uploader(
        "Upload filtered Excel",
        type=["xlsx"],
        key="step2_upload_excel",
    )

    if uploaded_file:
        full_df = pd.read_excel(uploaded_file)
        full_df = normalize_selected_column(full_df)

        st.markdown("### 🔘 Global Selection Controls")
        colA, colB = st.columns(2)

        if "upload_full_df" not in st.session_state:
            st.session_state["upload_full_df"] = full_df.copy()

        st.session_state["upload_full_df"] = normalize_selected_column(full_df)

        with colA:
            if st.button("✅ Select All (Upload)", key="upload_select_all"):
                st.session_state["upload_full_df"]["Selected"] = True

        with colB:
            if st.button("❌ Clear All (Upload)", key="upload_clear_all"):
                st.session_state["upload_full_df"]["Selected"] = False

        st.info("Uploaded file loaded. You can modify selection below.")

        edited_df = st.data_editor(
            st.session_state["upload_full_df"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Selected": st.column_config.CheckboxColumn(
                    "Selected",
                    default=False,
                )
            },
            disabled=[col for col in st.session_state["upload_full_df"].columns if col != "Selected"],
            key="uploaded_editor",
        )

        st.session_state["upload_full_df"] = edited_df.copy()

        selected_df = (
            edited_df[edited_df["Selected"] == True]
            .drop(columns=["Selected"])
            .reset_index(drop=True)
        )

# ---------------- FROM STEP 1 ----------------
if source_option == "From Step 1":
    if "step1_df" not in st.session_state:
        st.warning("Run Step 1 first.")
    else:
        full_df, selected_df = step2_filter_ui(st.session_state["step1_df"])

# ---------------- PREVIEW + COMMIT ----------------
if selected_df is not None and not selected_df.empty:
    st.subheader("Final Papers Going to Step 3")
    st.dataframe(selected_df, use_container_width=True)
    st.success(f"{len(selected_df)} papers selected.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Use Selected Papers → Step 3", key="commit_step2_to_step3"):
            st.session_state["step2_df"] = selected_df.copy()

            path = os.path.join(FILTER_DIR, "step2_selected_results.xlsx")
            selected_df.to_excel(path, index=False)

            st.success("Selected papers saved and forwarded to Step 3.")

    with col2:
        buffer = io.BytesIO()
        selected_df.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "⬇ Download Selected Papers (Excel)",
            data=buffer,
            file_name="step2_selected_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_step2_selected_excel",
        )

elif selected_df is not None:
    st.warning("⚠ No papers selected yet. Please tick 'Selected' to proceed.")

st.divider()


# =====================================================
# STEP 3 — PDF DOWNLOAD
# =====================================================
st.header("Step 3 — Download PDFs")

if "step2_df" not in st.session_state:
    st.warning("No filtered dataset available.")
else:
    step3_full_df = st.session_state["step2_df"].copy()
    step3_full_df["__paper_key"] = step3_full_df.apply(get_paper_key, axis=1)

    existing_step3_keys = set(st.session_state["step3_pdf_map"].keys())

    already_done_df = step3_full_df[step3_full_df["__paper_key"].isin(existing_step3_keys)].copy()
    pending_step3_df = step3_full_df[~step3_full_df["__paper_key"].isin(existing_step3_keys)].copy()

    st.dataframe(
        step3_full_df.drop(columns=["__paper_key"], errors="ignore"),
        use_container_width=True
    )

    st.info(
        f"Total selected papers: {len(step3_full_df)} | "
        f"Already processed in this session: {len(already_done_df)} | "
        f"Pending new downloads: {len(pending_step3_df)}"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("📥 Download PDFs", key="download_step3_pdfs", disabled=len(pending_step3_df) == 0):
            with st.spinner("Downloading only new PDFs..."):
                pdf_paths, report_df = download_pdfs(
                    pending_step3_df.drop(columns=["__paper_key"], errors="ignore"),
                    output_dir=PDF_DIR,
                    report_path="outputs/pdf_download_report.xlsx",
                )

                pending_keys = pending_step3_df["__paper_key"].tolist()

                for idx, pdf_path in enumerate(pdf_paths):
                    if idx >= len(pending_keys):
                        break

                    paper_key = pending_keys[idx]
                    row = pending_step3_df[pending_step3_df["__paper_key"] == paper_key].iloc[0]

                    st.session_state["step3_pdf_map"][paper_key] = {
                        "paper_key": paper_key,
                        "paper_title": row.get("Paper Title"),
                        "doi": row.get("DOI"),
                        "pdf_path": pdf_path,
                    }

                st.session_state["download_report_df"] = report_df
                rebuild_downloaded_pdfs_from_map()

                st.success(f"Added {len(pdf_paths)} new PDF(s).")

    with col_b:
        if st.button("🔄 Reset Step 3 Session PDF Tracking", key="reset_step3_tracking"):
            st.session_state["step3_pdf_map"] = {}
            st.session_state["downloaded_pdfs"] = []
            st.session_state["download_report_df"] = pd.DataFrame()
            st.success("Step 3 session PDF tracking cleared.")

    all_downloaded_pdfs = get_step3_all_pdf_paths()

    if all_downloaded_pdfs:
        st.success(f"{len(all_downloaded_pdfs)} total PDF(s) available in this session.")

        files_for_zip = load_step3_pdfs_for_zip(all_downloaded_pdfs)

        if "download_report_df" in st.session_state and isinstance(st.session_state["download_report_df"], pd.DataFrame) and not st.session_state["download_report_df"].empty:
            excel_buffer = io.BytesIO()
            st.session_state["download_report_df"].to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            files_for_zip["pdf_download_report.xlsx"] = excel_buffer.read()

        zip_buffer = create_zip(files_for_zip)

        st.download_button(
            "⬇ Download PDFs + Report (ZIP)",
            data=zip_buffer,
            file_name="pdfs_and_report.zip",
            mime="application/zip",
            key="zip_download",
        )

        if "download_report_df" in st.session_state and isinstance(st.session_state["download_report_df"], pd.DataFrame) and not st.session_state["download_report_df"].empty:
            excel_buffer = io.BytesIO()
            st.session_state["download_report_df"].to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)

            st.download_button(
                "⬇ Download Download Report (Excel)",
                data=excel_buffer,
                file_name="pdf_download_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download",
            )

st.divider()


# =====================================================
# STEP 4 — PDF SUMMARIZATION
# =====================================================
st.header("Step 4 — Generate Technical Summaries")

pdf_source = st.radio(
    "Select PDF Source",
    ["From Step 3 Downloads", "Upload PDFs"],
    horizontal=True,
    key="step4_pdf_source",
)

all_pdf_inputs = []

# ---------------- SOURCE: STEP 3 DOWNLOADS ----------------
if pdf_source == "From Step 3 Downloads":
    session_pdf_paths = get_step3_all_pdf_paths()
    if session_pdf_paths:
        all_pdf_inputs = load_step3_pdfs_as_inputs(session_pdf_paths)

# ---------------- SOURCE: MANUAL UPLOAD ----------------
else:
    uploaded_pdfs = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="step4_uploaded_pdfs",
    )
    if uploaded_pdfs:
        all_pdf_inputs = uploaded_pdfs

# ---------------- STEP 4 SETTINGS ----------------
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox(
        "Model",
        options=[DEFAULT_MODEL, FALLBACK_MODEL],
        index=0,
        help=(
            "Use gemini-2.5-flash for better summary quality. "
            "Use gemini-2.5-flash-lite for lower cost / higher throughput."
        ),
        key="step4_model_name",
    )

with col2:
    pause_seconds = st.slider(
        "Pause between files (seconds)",
        min_value=0.0,
        max_value=15.0,
        value=8.0,
        step=0.5,
        help="Useful when processing multiple PDFs on rate-limited setups.",
        key="step4_pause_seconds",
    )

# ---------------- STATUS ----------------
if not all_pdf_inputs:
    st.warning("No PDFs available for summarization.")
else:
    existing_summary_keys = set(st.session_state["step4_summary_map"].keys())

    already_done_pdfs = [
        pdf for pdf in all_pdf_inputs
        if getattr(pdf, "name", "") in existing_summary_keys
    ]
    pending_pdf_inputs = [
        pdf for pdf in all_pdf_inputs
        if getattr(pdf, "name", "") not in existing_summary_keys
    ]

    st.success(f"{len(all_pdf_inputs)} PDF(s) available for this source.")
    st.info(
        f"Already summarized in this session: {len(already_done_pdfs)} | "
        f"Pending new summaries: {len(pending_pdf_inputs)}"
    )

    col_x, col_y = st.columns(2)

    with col_x:
        if st.button("🧠 Generate Summaries", key="generate_step4_summaries", disabled=len(pending_pdf_inputs) == 0):
            with st.spinner("Generating summaries only for new PDFs..."):
                new_summaries = process_papers(
                    uploaded_files=pending_pdf_inputs,
                    model_name=model_name,
                    pause_seconds=pause_seconds,
                )

                for pdf in pending_pdf_inputs:
                    pdf_name = getattr(pdf, "name", "")
                    summary_name = f"{os.path.splitext(pdf_name)[0]}_Summary.docx"

                    if summary_name in new_summaries:
                        file_bytes = new_summaries[summary_name]

                        out_path = os.path.join(SUMMARY_DIR, summary_name)
                        try:
                            with open(out_path, "wb") as f:
                                f.write(file_bytes)
                        except Exception as e:
                            st.error(f"Could not save summary file {summary_name}: {e}")

                        st.session_state["step4_summary_map"][pdf_name] = {
                            "source_pdf": pdf_name,
                            "summary_name": summary_name,
                            "summary_bytes": file_bytes,
                        }

                rebuild_summaries_from_map()
                st.success(f"Added {len(new_summaries)} new summary file(s).")

    with col_y:
        if st.button("🔄 Reset Step 4 Session Summary Tracking", key="reset_step4_tracking"):
            st.session_state["step4_summary_map"] = {}
            st.session_state["summaries"] = {}
            st.success("Step 4 session summary tracking cleared.")

    rebuild_summaries_from_map()

    if st.session_state.get("summaries"):
        st.markdown("### 📄 Generated Summaries")

        for fname, file_bytes in st.session_state["summaries"].items():
            st.download_button(
                label=f"⬇ Download {fname}",
                data=file_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"download_{fname}",
            )

        zip_buffer = create_zip_from_dict(st.session_state["summaries"])

        st.download_button(
            "⬇ Download All Summaries (ZIP)",
            data=zip_buffer,
            file_name="paper_summaries.zip",
            mime="application/zip",
            key="download_all_summaries_zip",
        )
