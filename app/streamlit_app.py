"""Contract Clause Risk Classifier — web application.

Run with: python -m streamlit run app/streamlit_app.py
"""

import sys
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.baseline_model import CATEGORIES, load_model, predict as predict_baseline
from src import risk_scoring
from src import explainability
from src import transformer_model
from src import document_processing
from src import report_generator

st.set_page_config(page_title="Contract Clause Risk Classifier", page_icon="📄", layout="wide")

RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}

CATEGORY_LABELS = {
    "Termination For Convenience": "Termination",
    "Renewal Term": "Renewal Term",
    "Notice Period To Terminate Renewal": "Renewal Notice Period",
    "Cap On Liability": "Liability Cap",
    "Uncapped Liability": "Uncapped Liability",
    "Non-Compete": "Non-Compete",
}

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2.2rem; max-width: 1080px;}

    .app-title {font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
    .app-subtitle {color: #9CA3AF; font-size: 0.95rem; margin-top: 2px; margin-bottom: 1.8rem;}

    .section-label {
        font-size: 0.76rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: #9CA3AF; margin: 1.2rem 0 0.5rem 0;
    }

    div.stButton > button {
        border-radius: 8px; font-weight: 600; padding: 0.55rem 1.5rem; border: none;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #DC2626, #B91C1C);
        box-shadow: 0 2px 8px rgba(220,38,38,0.3);
    }
    div.stButton > button[kind="primary"]:hover {box-shadow: 0 4px 14px rgba(220,38,38,0.45);}

    .risk-card {
        border-radius: 10px; padding: 16px 18px; margin-bottom: 12px;
        border-left: 4px solid #6B7280; background: rgba(255,255,255,0.03);
    }
    .risk-card.High {border-left-color: #EF4444; background: rgba(239,68,68,0.06);}
    .risk-card.Medium {border-left-color: #F59E0B; background: rgba(245,158,11,0.06);}
    .risk-card.Low {border-left-color: #10B981; background: rgba(16,185,129,0.06);}

    .risk-card .row {display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;}
    .risk-card .category {font-weight: 700; font-size: 1.0rem;}
    .risk-card .reason {color: #D1D5DB; font-size: 0.9rem; line-height: 1.45;}

    .badge {
        padding: 3px 12px; border-radius: 999px; font-size: 0.72rem;
        font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
    }
    .badge.High {background: rgba(239,68,68,0.18); color: #FCA5A5;}
    .badge.Medium {background: rgba(245,158,11,0.18); color: #FCD34D;}
    .badge.Low {background: rgba(16,185,129,0.18); color: #6EE7B7;}

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03); border-radius: 10px;
        padding: 12px 16px; border: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stFileUploaderDropzone"] {border-radius: 10px; border: 1.5px dashed rgba(255,255,255,0.16);}

    .info-block {
        background: rgba(255,255,255,0.03); border-radius: 10px; padding: 14px 16px;
        margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.06);
    }
    .info-block .title {font-weight: 700; font-size: 0.85rem; margin-bottom: 6px;}
    .info-block .body {font-size: 0.82rem; color: #9CA3AF; line-height: 1.5;}

    .chip {
        display: inline-block; background: rgba(255,255,255,0.06); border-radius: 6px;
        padding: 3px 9px; font-size: 0.74rem; margin: 2px 4px 2px 0; color: #D1D5DB;
    }

    .empty-state {
        text-align: center; padding: 3rem 1rem; color: #6B7280;
        border: 1.5px dashed rgba(255,255,255,0.12); border-radius: 12px;
    }
    .empty-state .icon {font-size: 2.2rem; margin-bottom: 0.5rem;}

    .footer-note {color: #6B7280; font-size: 0.78rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);}

    .overall-risk {
        display: flex; align-items: center; justify-content: space-between;
        border-radius: 12px; padding: 18px 22px; margin: 1rem 0 1.4rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .overall-risk.High {background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.3);}
    .overall-risk.Medium {background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.3);}
    .overall-risk.Low {background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.3);}
    .overall-risk .label {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #9CA3AF; margin-bottom: 2px;}
    .overall-risk .value {font-size: 1.5rem; font-weight: 700;}
    .overall-risk.High .value {color: #FCA5A5;}
    .overall-risk.Medium .value {color: #FCD34D;}
    .overall-risk.Low .value {color: #6EE7B7;}

    .doc-block {
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
        border-left: 3px solid transparent; font-size: 0.9rem; line-height: 1.55;
        color: #D1D5DB; white-space: pre-wrap;
    }
    .doc-block.High {border-left-color: #EF4444; background: rgba(239,68,68,0.06);}
    .doc-block.Medium {border-left-color: #F59E0B; background: rgba(245,158,11,0.06);}
    .doc-block.Low {border-left-color: #10B981; background: rgba(16,185,129,0.06);}
    .doc-block.plain {color: #9CA3AF;}
    .doc-block .tag {
        display: block; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px;
    }
    .doc-block.High .tag {color: #FCA5A5;}
    .doc-block.Medium .tag {color: #FCD34D;}
    .doc-block.Low .tag {color: #6EE7B7;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_baseline_model():
    return load_model()


def classify(text: str, use_transformer: bool, vectorizer, clf):
    if use_transformer:
        preds, token_weights = transformer_model.predict(text)
        categories = [c for c, _ in preds]
        terms = [t for t, _ in explainability.from_attention(token_weights)]
    else:
        preds = predict_baseline(text, vectorizer, clf)
        categories = [c for c, _ in preds]
        terms = [t for t, _ in explainability.from_tfidf(text, vectorizer)]
    return categories, terms


def render_risk_card(category: str, level: str, reason: str):
    label = CATEGORY_LABELS.get(category, category)
    st.markdown(f"""
    <div class="risk-card {level}">
        <div class="row">
            <span class="category">{label}</span>
            <span class="badge {level}">{level} risk</span>
        </div>
        <div class="reason">{reason}</div>
    </div>
    """, unsafe_allow_html=True)


def render_overall_risk(overall: str, flagged_count: int, total_sections: int):
    if overall == "None":
        return
    st.markdown(f"""
    <div class="overall-risk {overall}">
        <div>
            <div class="label">Overall Document Risk</div>
            <div class="value">{overall}</div>
        </div>
        <div style="text-align:right; color:#9CA3AF; font-size:0.85rem;">
            {flagged_count} of {total_sections} sections flagged
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_highlighted_document(clauses: list, results: list):
    """Renders the full document in original order, highlighting sections
    that were flagged with their highest risk level."""
    by_text = {}
    for r in results:
        by_text.setdefault(r["full_text"], []).append(r)

    for clause in clauses:
        matches = by_text.get(clause)
        if not matches:
            st.markdown(f'<div class="doc-block plain">{clause}</div>', unsafe_allow_html=True)
            continue

        top_level = min(matches, key=lambda r: RISK_ORDER[r["level"]])["level"]
        labels = ", ".join(CATEGORY_LABELS.get(m["category"], m["category"]) for m in matches)
        st.markdown(
            f'<div class="doc-block {top_level}"><span class="tag">{labels} · {top_level} risk</span>{clause}</div>',
            unsafe_allow_html=True,
        )


st.markdown('<div class="app-title">Contract Clause Risk Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Identify clause types and risk levels in commercial contracts</div>',
    unsafe_allow_html=True,
)

transformer_ready = transformer_model.is_available()

with st.sidebar:
    st.markdown('<div class="info-block">', unsafe_allow_html=True)
    st.markdown('<div class="title">Model</div>', unsafe_allow_html=True)
    model_options = ["Standard"]
    if transformer_ready:
        model_options.append("Enhanced (Transformer)")
    model_choice = st.selectbox("Model", model_options, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-block">', unsafe_allow_html=True)
    st.markdown('<div class="title">Categories Covered</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="chip">{CATEGORY_LABELS[c]}</span>' for c in CATEGORIES)
    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-block">', unsafe_allow_html=True)
    st.markdown('<div class="title">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="body">Risk levels combine model predictions with automated '
        'checks against clause wording. This tool is informational only and does '
        'not substitute for legal advice.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

vectorizer, clf = get_baseline_model()
use_transformer = model_choice.startswith("Enhanced")

tab_document, tab_clause = st.tabs(["Document", "Single Clause"])

with tab_document:
    uploaded_file = st.file_uploader(
        "Upload a contract",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.session_state.pop("document_results", None)
        st.markdown(
            '<div class="empty-state"><div class="icon">📄</div>'
            'Upload a PDF, DOCX, or TXT contract to begin<br>'
            '<span style="font-size:0.85rem;">The document will be split into clauses and analyzed automatically</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Reading document..."):
                raw_text = document_processing.extract_text(uploaded_file)
        except Exception:
            st.error("This file couldn't be read. Please check the format and try again.")
            raw_text = None

        if raw_text is not None:
            clauses = document_processing.split_into_clauses(raw_text)

            if not clauses:
                st.warning("No clause-sized sections were found in this document.")
            else:
                col_info, col_action = st.columns([3, 1])
                with col_info:
                    st.markdown(
                        f'<div class="section-label">{uploaded_file.name} · {len(clauses)} sections</div>',
                        unsafe_allow_html=True,
                    )
                with col_action:
                    run_analysis = st.button("Analyze", type="primary", use_container_width=True)

                if run_analysis:
                    progress = st.progress(0.0)
                    results = []
                    for i, clause in enumerate(clauses):
                        categories, _ = classify(clause, use_transformer, vectorizer, clf)
                        if categories:
                            for r in risk_scoring.assess(clause, categories):
                                results.append({
                                    "excerpt": clause[:160] + ("…" if len(clause) > 160 else ""),
                                    "full_text": clause,
                                    "category": r["category"],
                                    "level": r["level"],
                                    "reason": r["reason"],
                                })
                        progress.progress((i + 1) / len(clauses))
                    progress.empty()
                    st.session_state["document_results"] = results

                if "document_results" in st.session_state:
                    results = st.session_state["document_results"]

                    if not results:
                        st.info("No clauses matched the categories this tool covers.")
                    else:
                        overall = report_generator.compute_overall_risk(results)
                        breakdown = report_generator.risk_breakdown(results)
                        render_overall_risk(overall, len(results), len(clauses))

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Flagged clauses", len(results))
                        m2.metric("High risk", breakdown["High"])
                        m3.metric("Medium risk", breakdown["Medium"])
                        m4.metric("Low risk", breakdown["Low"])

                        col_view, col_csv, col_pdf = st.columns([2, 1, 1])
                        with col_view:
                            view_mode = st.radio(
                                "View", ["Findings List", "Highlighted Document"],
                                horizontal=True, label_visibility="collapsed",
                            )
                        with col_csv:
                            st.download_button(
                                "Download CSV",
                                data=report_generator.build_csv(uploaded_file.name, results),
                                file_name=f"{uploaded_file.name}_risk_report.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with col_pdf:
                            st.download_button(
                                "Download PDF",
                                data=report_generator.build_pdf(uploaded_file.name, results, CATEGORY_LABELS),
                                file_name=f"{uploaded_file.name}_risk_report.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )

                        st.markdown('<div class="section-label">Findings</div>', unsafe_allow_html=True)

                        if view_mode == "Findings List":
                            risk_filter = st.multiselect(
                                "Filter", ["High", "Medium", "Low"],
                                default=["High", "Medium", "Low"],
                                label_visibility="collapsed",
                            )
                            filtered = sorted(
                                [r for r in results if r["level"] in risk_filter],
                                key=lambda r: RISK_ORDER[r["level"]],
                            )
                            for r in filtered:
                                render_risk_card(r["category"], r["level"], r["reason"])
                                with st.expander("View full clause"):
                                    st.write(r["full_text"])
                        else:
                            render_highlighted_document(clauses, results)

with tab_clause:
    clause_text = st.text_area(
        "Clause text",
        height=150,
        placeholder="Paste a contract clause to classify it and assess its risk level…",
        label_visibility="collapsed",
    )

    _, col_btn = st.columns([4, 1])
    with col_btn:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True, key="clause_analyze")

    if analyze_clicked:
        if not clause_text.strip():
            st.warning("Enter a clause to analyze.")
        else:
            categories, terms = classify(clause_text, use_transformer, vectorizer, clf)

            if not categories:
                st.info("No matching category was detected for this clause.")
            else:
                st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
                for r in risk_scoring.assess(clause_text, categories):
                    render_risk_card(r["category"], r["level"], r["reason"])

                if terms:
                    st.markdown('<div class="section-label">Key Terms</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip">{t}</span>' for t in terms)
                    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Risk levels are generated automatically and are '
    'informational only. Consult a qualified professional before making decisions '
    'based on contract terms.</div>',
    unsafe_allow_html=True,
)
