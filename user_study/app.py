import streamlit as st 
from pathlib import Path 
from utils import load_json

#### CSS #### 
background_color = "#FFFFFF"
secondary_background_color = "#F0F2F6"
text_color = "#181818"
secondary_link_color = "#6c757d"
table_color = "#ffffff"
cite_color = "#2A2AC8"

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
.evidence-container {{
    background-color: {secondary_background_color};  
    height: calc(100vh - 100px);
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
}}
.statement-header {{
    margin-left: 40px;
    margin-right: 40px;
    margin-top: 10px;
    font-style: italic;
}}
.cite-str {{
    font-weight: 600;
    color: {cite_color}
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

def process_text(text):
    """From https://github.com/THUDM/LongCite/blob/main/demo.py"""
    special_char={
        '&': '&amp;',
        '\'': '&apos;',
        '"': '&quot;',
        '<': '&lt;',
        '>': '&gt;',
        '\n': '<br>',
        '$': '&#36;',
    }
    for x, y in special_char.items():
        text = text.replace(x, y)
    return text

def get_mapping_spans2cite_strs(results):

    cite_strs = []
    for sc in results["statements"]:
        citation = sc["citation"]

        sent_cite_strs = []
        if citation:
            spans = [c["span"] for c in citation]
            for span in spans:
                start, end = span
                if start == end:
                    cite_str = f"[{start}]"
                else:
                    cite_str = f"[{start}-{end}]"
                sent_cite_strs.append(cite_str)

        cite_strs.extend(sent_cite_strs)

    span2citestr = dict.fromkeys(cite_strs) 
    for i, key in enumerate(span2citestr, start=1):
        span2citestr[key] = f"[{i}]"

    return span2citestr

def render_sidebar():
    app_path = Path(__file__).resolve()
    cc_metrics_results_dir = app_path.parent.parent / "context_cite_metrics/results"
    eval_metrics_results_dir = app_path.parent.parent / "context_cite_metrics/results_final/averitec_short_ans"

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

def render_claim(results, col):

    # claim
    with col:
        st.markdown(f"""
                    <div class="claim">
                        <span class="section-label">Claim:</span> {results.get("claim")}
                    </div>
        """, unsafe_allow_html=True) 

def render_verdict(results, col):

    LABELS = ['Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking']
    COLORS = ["#28a745", "#cc1414", "#2995bd", "#fac104"]
    EMOTES = ["\u2705", "\u274C", "\u2754", "\u26A1"]

    pred_label = results.get("pred_label")

    # predicted verdict
    with col:
        idx = LABELS.index(pred_label)
        st.markdown(f"""
        <div class="verdict" style="background-color: {COLORS[idx]};">
            Predicted verdict: {pred_label} {EMOTES[idx]}
        </div>
        """, unsafe_allow_html=True)

def render_model_answer(results, span2citestr, col, include_citations=True):

    def build_answer_text(results, include_citations):
        answer_text = "\n"
        for sc in results["statements"]:
        
            statement = sc["statement"]
            answer_text += f"<span>{process_text(statement)}</span>"
            answer_text += " "
            
            if include_citations:
                citation = sc["citation"]
                sent_citation_strs = []
                if citation:
                    spans = [c["span"] for c in citation]
                    for span in spans:
                        start, end = span
                        if start == end:
                            cite_str = f"[{start}]"
                        else:
                            cite_str = f"[{start}-{end}]"
                        sent_citation_strs.append(cite_str)

                for cite_str in sent_citation_strs:
                    answer_text += f"<span class='cite-str'>{span2citestr.get(cite_str)}</span>"
                answer_text += "\n\n"

        return answer_text
    
    answer_text = build_answer_text(results, include_citations=include_citations)
    with col:
        st.markdown(f"""
            <div class="justification" id="justification">
                <span class="section-label">Justification:</span> 
                {answer_text}
        """, unsafe_allow_html=True)  # why on earth?

def render_evidences(results, span2citestr, col_r):

    with col_r:
        st.markdown(f"""
            <span class="evidence-heading">
                Evidence snippets:
            </span>
            <span style='font-size: 0.9rem;'>The actual content cited by the AI-system is displayed in <span style='font-weight:600;'>bold</span>. Refer to the additional text if you need more context.</span>
        """, unsafe_allow_html=True) 

    statements = results["statements"]

    # build evidence html
    evidence_html = ""
    for i in range(len(statements)):
        sc = statements[i]
        citation = sc["citation"]

        if citation:
            evidence_html += f"""
                <div class="statement-header">
                    Statement {i+1}:
                </div>
            """    
        for c in citation:
            cite_text, pre_context, post_context, cite_span = (
                process_text(c["cite"]), 
                process_text(c["pre_context"]), 
                process_text(c["post_context"]), 
                c["span"],
            )
            start, end = cite_span 
            if start == end:
                cite_str = f"[{start}]"
            else:
                cite_str = f"[{start}-{end}]"

            evidence_html += f"""
            <div class="evidence-card">
                <div class="evidence-title">
                    <span>Evidence snippet </span>
                    <span style="color: {cite_color};">{span2citestr.get(cite_str)}</span>
                </div>
                <div class="evidence-content">
                    <span style='font-weight: 300;'>{pre_context}</span>
                    <span style='font-weight: 600;'>{cite_text}</span>
                    <span style='font-weight: 300;'>{post_context}</span>
                </div>
            </div>
        """

    with col_r:
        st.html(f"<div class='evidence-container'>{evidence_html}</div>")
            
def main():
    # read query parameters for which claim and which experiment group to display
    query_params = st.query_params
    exp_group = query_params["exp_group"]
    idx = int(query_params["item"]) - 1  # item query param starts at 1

    # load corresponding results 
    if exp_group == "A":
        results_path = "./user_study_data/data_faithful.json"
    elif exp_group == "B":
        results_path = "./user_study_data/data_unfaithful.json"
    elif exp_group == "C":
        results_path = "./user_study_data/data_faithful.json"  # does not matter since no citations are displayed
    results = load_json(results_path)[idx]

    # with citations
    if exp_group == "A" or exp_group == "B":
        st.set_page_config(layout="wide")

        # layout: two columns left and right
        col_l, col_r = st.columns([1,1])

        # mapping to simplify the cite_strs to start at 1,2,3,...
        span2citestr = get_mapping_spans2cite_strs(results)

        # display content
        render_claim(results, col_l)
        render_verdict(results, col_l)
        render_model_answer(results, span2citestr, col_l)
        render_evidences(results, span2citestr, col_r)

    # no citations
    elif exp_group == "C":
        st.set_page_config(layout="centered")

        # layout: only single column in the center
        col_center = st.container() 

        # display content 
        render_claim(results, col_center)
        render_verdict(results, col_center)
        render_model_answer(results, None, col_center, include_citations=False)
        
    

if __name__ == "__main__":
    main()