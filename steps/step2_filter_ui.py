import streamlit as st
import pandas as pd


def step2_filter_ui(df: pd.DataFrame):
    st.subheader("Filter & Select Papers")

    # ---------------- Preserve Original Data ----------------
    if "original_results" not in st.session_state:
        st.session_state["original_results"] = df.copy()

    working_df = st.session_state["original_results"].copy()
    original_count = len(working_df)

    # ---------------- Filters ----------------
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        min_citations = st.number_input("Min citations", min_value=0, value=0)

    with col2:
        reviews_only = st.checkbox("Reviews only")

    with col3:
        open_access_only = st.checkbox("Open access only", value=True)

    with col4:
        top_n = st.number_input("Top N (0 = all)", min_value=0, value=0)

    with col5:
        min_Relevance_Score = st.number_input("min Relevance Score", min_value=0, value=0)

    # ---------------- Apply Filters ----------------
    filtered_df = working_df.copy()

    if "Citations Count" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Citations Count"] >= min_citations
        ]

    if reviews_only and "Review" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Review"] == "YES"
        ]

    if open_access_only and "Open Access" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Open Access"] == True
        ]

    if top_n > 0 and "Citations Count" in filtered_df.columns:
        filtered_df = (
            filtered_df
            .sort_values("Citations Count", ascending=False)
            .head(top_n)
        )
    
    if "Relevance Score" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Relevance Score"] >= min_Relevance_Score
        ]

    filtered_df = filtered_df.reset_index(drop=True)
    filtered_count = len(filtered_df)

    # ---------------- Ensure Selected Column ----------------
    if "Selected" not in filtered_df.columns:
        filtered_df.insert(0, "Selected", False)

    # ---------------- GLOBAL SELECT CONTROLS ----------------
    st.markdown("### 🔘 Global Selection Controls")

    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        if st.button("✅ Select All (Filtered)"):
            filtered_df["Selected"] = True

    with colB:
        if st.button("❌ Clear All (Filtered)"):
            filtered_df["Selected"] = False

    with colC:
        st.info(f"Filtered Results: {filtered_count}")

    # ---------------- Editable Table ----------------
    edited_df = st.data_editor(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Selected": st.column_config.CheckboxColumn(
                "Selected",
                help="Tick to include paper in next step",
                default=False
            )
        },
        disabled=[col for col in filtered_df.columns if col != "Selected"],
        key="paper_selector_editor"
    )

    # ---------------- Extract Selected ----------------
    selected_df = edited_df[edited_df["Selected"] == True] \
        .drop(columns=["Selected"]) \
        .reset_index(drop=True)

    st.success(f"Selected for next step: {len(selected_df)}")

    return edited_df, selected_df