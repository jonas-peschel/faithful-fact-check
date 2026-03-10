import streamlit as st 
from pathlib import Path 
from utils import load_json

#--- citation recall colors ---#
SUPPORTED_COLOR = "#c8f7c5"
PARTIAL_SUPPORT_COLOR = "#fdf7c3"
UNSUPPORTED_COLOR = "#f7c5c5"
SUPPORT_SCORE2COLOR = {
    0: UNSUPPORTED_COLOR,
    0.5: PARTIAL_SUPPORT_COLOR,
    1: SUPPORTED_COLOR,
}

#--- citation precision colors ---#
RELEVANT_COLOR = "#1b8d36"
IRRELEVANT_COLOR = "#cc1414"
RELEVANT_SCORE2COLOR = {
    0: IRRELEVANT_COLOR,
    1: RELEVANT_COLOR,
}

#### CSS #### 
background_color = "#FFFFFF"
secondary_background_color = "#F0F2F6"
text_color = "#181818"
secondary_link_color = "#6c757d"

# CSS for page layout
css_layout = f"""
.claim {{
    margin: 20px 40px;
    background-color: {background_color};
    padding: 1.5rem;
    font-size: 1.1rem;
}}
.section-label {{
    font-weight: bold;
}}
.evidence-heading{{
    font-weight: bold;
    font-size: 1.1rem;
    margin-bottom: 10px;
}}
.verdict {{
    color: white; 
    padding: 10px; 
    border-radius: 5px; 
    text-align: center; 
    font-weight: bold;
    margin: 10px 40px;
}}
.justification {{
    margin: 20px 40px 50px 40px;
    background-color: {background_color};
    padding: 1.5rem;
}}
.st-key-evidence-container {{
    background-color: {secondary_background_color};  
    height: calc(100vh - 150px);
    overflow-y: auto;
    position: sticky;
    top: 0;
}}
.evidence-card {{
    background-color: {background_color};
    padding: 1.5rem;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
    margin-left: 40px;
    margin-right: 40px;
}}

.evidence-title {{
    font-weight: 600;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.statement-header {{
    margin-left: 40px;
    margin-right: 40px;
    margin-top: 10px;
    font-style: italic;
}}
"""


# CSS for highlighting the source sentence to which the user is linked by clicking on tooltip
css_highlighting = f"""

:target:not(#justification) {{
    animation: hightlight-flash 0.75s linear;
}}

@keyframes hightlight-flash {{
    0% {{
        background-color: {secondary_link_color};
        opacity: 0.1;
    }}
    50% {{
        background-color: {secondary_link_color};
        opacity: 0.3;
    }}
    100% {{
        background-color: {background_color};
    }}
}}

"""

st.html(f"<style>{css_layout}</style>")
st.html(f"<style>{css_highlighting}</style>")
#### CSS END #### 

# from https://github.com/THUDM/LongCite/blob/main/demo.py
def process_text(text):
    special_char={
        '&': '&amp;',
        '\'': '&apos;',
        '"': '&quot;',
        '<': '&lt;',
        '>': '&gt;',
        '\n': '<br>',
    }
    for x, y in special_char.items():
        text = text.replace(x, y)
    return text

def render_sidebar():
    app_path = Path(__file__).resolve()
    cc_metrics_results_dir = app_path.parent.parent / "context_cite_metrics/results"
    eval_metrics_results_dir = app_path.parent.parent / "context_cite_metrics/results_final/AVeriTeC"

    with st.sidebar:
        st.subheader("Settings")

        # ContextCite metrics results folder 
        cc_metrics_results_path = None
        cc_metrics_results_file = st.selectbox(
            label="ContextCite metrics results file",
            options=[path.name for path in cc_metrics_results_dir.iterdir() if path.is_file()],
            index=None,
            placeholder="ContextCite metrics results file", 
        )
        if cc_metrics_results_file:
            cc_metrics_results_path = cc_metrics_results_dir / cc_metrics_results_file

        # eval metrics results folder 
        eval_metrics_results_path = None
        eval_metrics_results_file = st.selectbox(
            label="Evaluation metrics results file",
            options=[path.name for path in eval_metrics_results_dir.iterdir() if path.is_file()],
            index=None,
            placeholder="Evaluation metrics results file", 
        )
        if eval_metrics_results_file:
            eval_metrics_results_path = eval_metrics_results_dir / eval_metrics_results_file

        idx = None
        idx = st.number_input(
            label="Claim Index",
            value=0,
            min_value=0,
            help="Claim of which the results should be displayed",
            placeholder="Claim Index",
            step=1,
            format="%d"
        )

    return cc_metrics_results_path, eval_metrics_results_path, idx

def render_claim(results, col_l):

    # claim
    with col_l:
        st.markdown(f"""
                    <div class="claim">
                        <span class="section-label">Claim:</span> {results.get("claim")}
                    </div>
        """, unsafe_allow_html=True) 

def render_verdicts(results, col_l):

    LABELS = ['Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking']
    COLORS = ["#28a745", "#cc1414", "#2995bd", "#fac104"]
    EMOTES = ["\u2705", "\u274C", "\u2754", "\u26A1"]

    gt_label = results.get("label")
    pred_label = results.get("pred_label")

    # predicted verdict
    with col_l:
        idx = LABELS.index(pred_label)
        st.markdown(f"""
        <div class="verdict" style="background-color: {COLORS[idx]};">
            Predicted verdict: {pred_label} {EMOTES[idx]}
        </div>
        """, unsafe_allow_html=True)

    # ground truth
    idx = LABELS.index(gt_label)
    with col_l:
        st.markdown(f"""
        <div class="verdict" style="background-color: {COLORS[idx]};">
            Ground truth verdict: {gt_label} {EMOTES[idx]}
        </div>
        """, unsafe_allow_html=True)

def render_model_answer(results, col_l):

    def build_answer_text(results):

        answer_text = "\n"
        for sc in results["statements"]:
            statement = sc["statement"]
            citation = sc["citation"]
            support_score = sc["support_score"]
            relevant_scores = []

            if citation:
                spans = [c["span"] for c in citation]
                relevant_scores = [c["relevant_score"] for c in citation]
                sent_citation_strs = []
                for span in spans:
                    start, end = span
                    if start == end:
                        cite_str = f"[{start}]"
                    else:
                        cite_str = f"[{start}-{end}]"
                    sent_citation_strs.append(cite_str)
            else:
                sent_citation_strs = [""]
        
            # apply background color depending on whether the statement is supported or not
            color = SUPPORT_SCORE2COLOR[support_score]
            answer_text += f"<span style='background-color: {color};'>{statement}</span>"
            answer_text += " "

            # apply color to the citation depending on whether it is relevant or not
            cite_colors = [RELEVANT_SCORE2COLOR[relevant_score] for relevant_score in relevant_scores]
            for cite_str, col in zip(sent_citation_strs, cite_colors):
                answer_text += f"<span style='color: {col};'>{cite_str}</span>"

            answer_text += "\n\n"

        return answer_text
    

    answer_text = build_answer_text(results)
    with col_l:
        st.markdown(f"""
            <div class="justification" id="justification">
                <span class="section-label">Justification:</span> 
                {answer_text}
        """, unsafe_allow_html=True)  # why on earth?

def render_evidences(eval_metrics_results, col_r):

    with col_r:
        st.markdown(f"""
                    <div class="evidence-heading">
                        Evidence snippets:
                    </div>
        """, unsafe_allow_html=True) 

    with col_r:
        container = st.container(key="evidence-container", border=False)

    # build evidence html
    evidence_html = ""
    for i, sc in enumerate(eval_metrics_results["statements"], start=1):
        citation = sc["citation"]
        if citation:
            evidence_html += f"""
                <div class="statement-header">
                    Statement {i}:
                </div>
            """    
        for c in citation:
            cite_text, cite_span, cite_score, relevant_score = process_text(c["cite"]), c["span"], c["score"], c["relevant_score"]
            start, end = cite_span 
            if start == end:
                cite_span_str = f"[{start}]"
            else:
                cite_span_str = f"[{start}-{end}]"

            if cite_score:        
                score_html = f"<span>Score: {cite_score:.2f}</span>"
            else:
                score_html = f"<span>Score: -</span>"

            relevant_text = "Relevant" if relevant_score == 1 else "Irrelevant"
            relevant_color = RELEVANT_SCORE2COLOR[relevant_score]

            evidence_html += f"""
                <div class="evidence-card">
                    <div class="evidence-title">
                        <span style='color: {relevant_color};'>{relevant_text}</span>
                        {score_html}
                    </div>
                    <div class="evidence-title">
                        <span>Evidence snippet {cite_span_str}</span>
                    </div>
                    <div class="evidence-content">{cite_text}</div>
                </div>
            """

    with container:
        st.markdown(evidence_html, unsafe_allow_html=True)
            
def main():
    st.set_page_config(layout="wide")
    st.title("Evaluation metrics visualization")

    # get settings
    cc_metrics_results_path, eval_metrics_results_path, idx = render_sidebar()
    eval_metrics_results = load_json(eval_metrics_results_path)[idx]

    col_l, col_r = st.columns([1,1])

    render_claim(eval_metrics_results, col_l)
    render_verdicts(eval_metrics_results, col_l)
    render_model_answer(eval_metrics_results, col_l)
    render_evidences(eval_metrics_results, col_r)
    


if __name__ == "__main__":
    main()