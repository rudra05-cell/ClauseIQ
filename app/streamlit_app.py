"""ClauseIQ — Contract Clause Risk Classifier web application.

Run with: python -m streamlit run app/streamlit_app.py
"""

import sys
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import joblib
from src.baseline_model import CATEGORIES, load_model, predict as predict_baseline
from src import risk_scoring
from src import explainability
from src import transformer_model
from src import document_processing
from src import report_generator
from src import suggestions
from src import feedback

st.set_page_config(page_title="ClauseIQ", page_icon="🧠", layout="wide")

RISK_ORDER = {"High": 0, "Medium": 1, "Low": 2}

CATEGORY_LABELS = {
    "Termination For Convenience": "Termination",
    "Renewal Term": "Renewal Term",
    "Notice Period To Terminate Renewal": "Renewal Notice Period",
    "Cap On Liability": "Liability Cap",
    "Uncapped Liability": "Uncapped Liability",
    "Non-Compete": "Non-Compete",
    "Audit Rights": "Audit Rights",
    "Insurance": "Insurance",
    "Exclusivity": "Exclusivity",
    "Warranty Duration": "Warranty Duration",
    "Volume Restriction": "Volume Restriction",
}

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2.2rem; max-width: 1100px;}

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

    .footer-note {color: #6B7280; font-size: 0.78rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_baseline_model():
    return load_model()


def classify(text: str, use_transformer: bool, vectorizer, clf):
    """Returns (categories, terms, confidence_by_category)."""
    if use_transformer:
        preds, token_weights = transformer_model.predict(text)
        categories = [c for c, _ in preds]
        confidence = {c: p for c, p in preds}
        terms = [t for t, _ in explainability.from_attention(token_weights)]
    else:
        preds = predict_baseline(text, vectorizer, clf)
        categories = [c for c, _ in preds]
        confidence = {c: p for c, p in preds}
        terms = [t for t, _ in explainability.from_tfidf(text, vectorizer)]
    return categories, terms, confidence


def render_risk_card(category: str, level: str, reason: str, confidence: float = None, key_prefix: str = None):
    label = CATEGORY_LABELS.get(category, category)
    confidence_html = (
        f'<span style="color:#9CA3AF; font-size:0.78rem; margin-left:8px;">{confidence*100:.0f}% confidence</span>'
        if confidence is not None else ""
    )
    suggestion = suggestions.suggest(category, level)
    suggestion_html = (
        f'<div class="reason" style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.08);">'
        f'<b>Suggestion:</b> {suggestion}</div>'
        if suggestion else ""
    )
    st.markdown(f"""
    <div class="risk-card {level}">
        <div class="row">
            <span class="category">{label}{confidence_html}</span>
            <span class="badge {level}">{level} risk</span>
        </div>
        <div class="reason">{reason}</div>
        {suggestion_html}
    </div>
    """, unsafe_allow_html=True)

    if key_prefix:
        fb_col1, fb_col2, _ = st.columns([1, 1, 6])
        with fb_col1:
            if st.button("Correct", key=f"{key_prefix}_correct", use_container_width=True):
                feedback.log_feedback("(see analysis above)", category, level, "correct")
                st.toast("Feedback recorded")
        with fb_col2:
            if st.button("Incorrect", key=f"{key_prefix}_incorrect", use_container_width=True):
                feedback.log_feedback("(see analysis above)", category, level, "incorrect")
                st.toast("Feedback recorded")


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


st.markdown('<div class="app-title">🧠 ClauseIQ</div>', unsafe_allow_html=True)
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

    fb_summary = feedback.load_feedback_summary()
    if fb_summary["total"] > 0:
        st.markdown('<div class="info-block">', unsafe_allow_html=True)
        st.markdown('<div class="title">Feedback Logged</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="body">{fb_summary["correct"]} correct, {fb_summary["incorrect"]} incorrect, '
            f'{fb_summary["total"]} total</div>',
            unsafe_allow_html=True,
        )
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

tab_document, tab_clause, tab_batch, tab_compare = st.tabs(
    ["Document", "Single Clause", "Batch", "Compare Clauses"]
)

# ============================================================================
# TAB: DOCUMENT
# ============================================================================
with tab_document:
    uploaded_file = st.file_uploader(
        "Upload a contract", type=["pdf", "docx", "txt"], label_visibility="collapsed", key="doc_uploader",
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
                        categories, _, confidence = classify(clause, use_transformer, vectorizer, clf)
                        if categories:
                            for r in risk_scoring.assess(clause, categories):
                                results.append({
                                    "excerpt": clause[:160] + ("…" if len(clause) > 160 else ""),
                                    "full_text": clause,
                                    "category": r["category"],
                                    "level": r["level"],
                                    "reason": r["reason"],
                                    "confidence": confidence.get(r["category"]),
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
                            for idx, r in enumerate(filtered):
                                render_risk_card(r["category"], r["level"], r["reason"], r.get("confidence"))
                                with st.expander("View full clause"):
                                    st.write(r["full_text"])
                        else:
                            render_highlighted_document(clauses, results)

# ============================================================================
# TAB: SINGLE CLAUSE
# ============================================================================
with tab_clause:
    clause_text = st.text_area(
        "Clause text", height=150,
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
            categories, terms, confidence = classify(clause_text, use_transformer, vectorizer, clf)

            if not categories:
                st.info("No matching category was detected for this clause.")
            else:
                st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
                for idx, r in enumerate(risk_scoring.assess(clause_text, categories)):
                    render_risk_card(
                        r["category"], r["level"], r["reason"], confidence.get(r["category"]),
                        key_prefix=f"single_{idx}",
                    )

                if terms:
                    st.markdown('<div class="section-label">Key Terms</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip">{t}</span>' for t in terms)
                    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

# ============================================================================
# TAB: BATCH
# ============================================================================
with tab_batch:
    st.markdown(
        '<div class="section-label">Upload multiple contracts for a portfolio-level view</div>',
        unsafe_allow_html=True,
    )
    batch_files = st.file_uploader(
        "Upload contracts", type=["pdf", "docx", "txt"], accept_multiple_files=True,
        label_visibility="collapsed", key="batch_uploader",
    )

    if not batch_files:
        st.markdown(
            '<div class="empty-state"><div class="icon">🗂️</div>'
            'Upload two or more contracts to see a portfolio risk summary'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        if st.button("Analyze All", type="primary"):
            batch_summary = []
            progress = st.progress(0.0)
            for i, bf in enumerate(batch_files):
                try:
                    text = document_processing.extract_text(bf)
                    clauses = document_processing.split_into_clauses(text)
                except Exception:
                    batch_summary.append({"name": bf.name, "overall": "Error", "high": 0, "medium": 0, "low": 0, "flagged": 0})
                    progress.progress((i + 1) / len(batch_files))
                    continue

                doc_results = []
                for c in clauses:
                    cats, _, _ = classify(c, use_transformer, vectorizer, clf)
                    if cats:
                        for r in risk_scoring.assess(c, cats):
                            doc_results.append(r)

                breakdown = report_generator.risk_breakdown(doc_results)
                batch_summary.append({
                    "name": bf.name,
                    "overall": report_generator.compute_overall_risk(doc_results),
                    "high": breakdown["High"], "medium": breakdown["Medium"], "low": breakdown["Low"],
                    "flagged": len(doc_results),
                })
                progress.progress((i + 1) / len(batch_files))
            progress.empty()
            st.session_state["batch_summary"] = batch_summary

        if "batch_summary" in st.session_state:
            st.markdown('<div class="section-label">Portfolio Summary</div>', unsafe_allow_html=True)
            summary = st.session_state["batch_summary"]

            high_count = sum(1 for s in summary if s["overall"] == "High")
            m1, m2, m3 = st.columns(3)
            m1.metric("Contracts analyzed", len(summary))
            m2.metric("High-risk contracts", high_count)
            m3.metric("Total flagged clauses", sum(s["flagged"] for s in summary))

            st.markdown('<div class="section-label">By Contract</div>', unsafe_allow_html=True)
            for s in sorted(summary, key=lambda x: RISK_ORDER.get(x["overall"], 3)):
                if s["overall"] == "Error":
                    st.markdown(f'<div class="risk-card"><div class="row"><span class="category">{s["name"]}</span>'
                                f'<span class="badge">Unreadable</span></div></div>', unsafe_allow_html=True)
                    continue
                st.markdown(f"""
                <div class="risk-card {s['overall']}">
                    <div class="row">
                        <span class="category">{s['name']}</span>
                        <span class="badge {s['overall']}">{s['overall']} risk</span>
                    </div>
                    <div class="reason">{s['flagged']} flagged sections — {s['high']} High, {s['medium']} Medium, {s['low']} Low</div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# TAB: COMPARE CLAUSES
# ============================================================================
def _analyze_side(label: str, key_prefix: str):
    """Lets the user paste text or upload a document for one side of the
    comparison, and returns a unified result list regardless of which input
    mode was used (a pasted clause is treated as a single-section document)."""
    input_mode = st.radio(
        f"{label} input", ["Paste text", "Upload document"],
        horizontal=True, label_visibility="collapsed", key=f"{key_prefix}_mode",
    )

    clauses = []
    if input_mode == "Paste text":
        text = st.text_area(label, height=150, label_visibility="collapsed", key=f"{key_prefix}_text")
        if text.strip():
            clauses = [text.strip()]
    else:
        uploaded = st.file_uploader(
            label, type=["pdf", "docx", "txt"], label_visibility="collapsed", key=f"{key_prefix}_file",
        )
        if uploaded is not None:
            try:
                raw_text = document_processing.extract_text(uploaded)
                clauses = document_processing.split_into_clauses(raw_text)
            except Exception:
                st.error("This file couldn't be read.")

    return clauses


def _evaluate_clauses(clauses: list):
    results = []
    for clause in clauses:
        cats, _, conf = classify(clause, use_transformer, vectorizer, clf)
        if cats:
            for r in risk_scoring.assess(clause, cats):
                results.append({
                    "category": r["category"], "level": r["level"], "reason": r["reason"],
                    "confidence": conf.get(r["category"]), "full_text": clause,
                })
    return results


with tab_compare:
    st.markdown(
        '<div class="section-label">Compare two clauses or two full documents (e.g. before/after negotiation, or two competing vendor contracts)</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-label">Version A</div>', unsafe_allow_html=True)
        clauses_a = _analyze_side("Version A", "compare_a")
    with col_b:
        st.markdown('<div class="section-label">Version B</div>', unsafe_allow_html=True)
        clauses_b = _analyze_side("Version B", "compare_b")

    _, col_btn = st.columns([4, 1])
    with col_btn:
        compare_clicked = st.button("Compare", type="primary", use_container_width=True, key="compare_analyze")

    if compare_clicked:
        if not clauses_a or not clauses_b:
            st.warning("Provide both Version A and Version B — paste text or upload a document for each.")
        else:
            results_a = _evaluate_clauses(clauses_a)
            results_b = _evaluate_clauses(clauses_b)
            overall_a = report_generator.compute_overall_risk(results_a)
            overall_b = report_generator.compute_overall_risk(results_b)
            breakdown_a = report_generator.risk_breakdown(results_a)
            breakdown_b = report_generator.risk_breakdown(results_b)

            st.markdown('<div class="section-label">Overall Comparison</div>', unsafe_allow_html=True)
            col_a2, col_b2 = st.columns(2)
            with col_a2:
                render_overall_risk(overall_a, len(results_a), len(clauses_a))
                st.caption(f"{breakdown_a['High']} High · {breakdown_a['Medium']} Medium · {breakdown_a['Low']} Low")
            with col_b2:
                render_overall_risk(overall_b, len(results_b), len(clauses_b))
                st.caption(f"{breakdown_b['High']} High · {breakdown_b['Medium']} Medium · {breakdown_b['Low']} Low")

            order = {"High": 0, "Medium": 1, "Low": 2, "None": 3}
            if order[overall_a] < order[overall_b]:
                st.info("Version A carries higher overall risk than Version B.")
            elif order[overall_b] < order[overall_a]:
                st.info("Version B carries higher overall risk than Version A.")
            else:
                st.info("Both versions carry a similar overall risk level.")

            cats_a = {r["category"] for r in results_a}
            cats_b = {r["category"] for r in results_b}
            only_in_a = cats_a - cats_b
            only_in_b = cats_b - cats_a
            shared = cats_a & cats_b

            if only_in_a or only_in_b or shared:
                st.markdown('<div class="section-label">Category Differences</div>', unsafe_allow_html=True)
                if only_in_a:
                    st.markdown(f"**Only in Version A:** {', '.join(CATEGORY_LABELS.get(c, c) for c in only_in_a)}")
                if only_in_b:
                    st.markdown(f"**Only in Version B:** {', '.join(CATEGORY_LABELS.get(c, c) for c in only_in_b)}")
                for cat in shared:
                    level_a = min((r["level"] for r in results_a if r["category"] == cat), key=lambda l: order[l])
                    level_b = min((r["level"] for r in results_b if r["category"] == cat), key=lambda l: order[l])
                    if level_a != level_b:
                        st.markdown(f"**{CATEGORY_LABELS.get(cat, cat)}:** {level_a} in A vs {level_b} in B")

            col_a3, col_b3 = st.columns(2)
            with col_a3:
                st.markdown("**Version A — findings**")
                if not results_a:
                    st.caption("No matching categories detected.")
                for r in sorted(results_a, key=lambda r: RISK_ORDER[r["level"]]):
                    render_risk_card(r["category"], r["level"], r["reason"], r.get("confidence"))
            with col_b3:
                st.markdown("**Version B — findings**")
                if not results_b:
                    st.caption("No matching categories detected.")
                for r in sorted(results_b, key=lambda r: RISK_ORDER[r["level"]]):
                    render_risk_card(r["category"], r["level"], r["reason"], r.get("confidence"))

st.markdown(
    '<div class="footer-note">Risk levels are generated automatically and are '
    'informational only. Consult a qualified professional before making decisions '
    'based on contract terms.</div>',
    unsafe_allow_html=True,
)
