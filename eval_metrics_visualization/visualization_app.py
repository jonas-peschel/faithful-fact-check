import streamlit as st 
from pathlib import Path 
import re
import math
import json 
from urllib.parse import urlparse, quote
from utils import load_json, get_attr_method, METH2LABEL, LABEL2METH, FEATURED_EXAMPLE_IDXS, get_order_with_featured_examples, process_text, get_mapping_spans2cite_strs, show_explanations, SUPPORTED_COLOR, PARTIAL_SUPPORT_COLOR, UNSUPPORTED_COLOR, SUPPORT_SCORE2COLOR, RELEVANT_COLOR, IRRELEVANT_COLOR, RELEVANT_SCORE2COLOR, load_json_from_hub

#### CSS #### 
background_color = "#FFFFFF"
secondary_background_color = "#F0F2F6"
text_color = "#181818"
secondary_link_color = "#6c757d"
table_color = "#ffffff"

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
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.evidence-content {{
    margin-bottom: 1rem;
}}
.source-url a {{
    font-size: 0.8rem;
    color: {secondary_link_color};
    text-decoration: none;
}}
.statement-header {{
    margin-left: 40px;
    margin-right: 40px;
    margin-top: 10px;
    font-weight: 500;
    font-size: 1.05rem;
}}

.citation-metrics {{
    margin: 0 40px 30px 40px;
    padding: 1rem 1.5rem;
    background-color: {background_color};
    border-left: 4px solid #d0d4dc;        /* sets it apart from the evidence cards */
}}
.citation-metrics-header {{
    font-weight: 600;
    margin-bottom: 0.6rem;
}}
.metrics-grid {{
    display: grid;
    grid-template-columns: max-content max-content 1fr;
    column-gap: 0.75rem;
    row-gap: 0.4rem;
    align-items: center;
}}
.metric-row {{ display: contents; }}         /* children become grid cells */
.metric-label {{ color: {secondary_link_color}; }}
.metric-value {{
    font-variant-numeric: tabular-nums;    /* digits line up */
    text-align: right;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    background-color: var(--metric-color);
}}
.metric-desc {{ font-size: 0.9rem; }}

.drops-table{{
    margin: 0 40px 1.5rem 40px;
}}
td, th{{
    padding: 6px 12px;
    border: 1px solid #ddd;
    background-color: {table_color};
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

def render_sidebar():
    n_claims = 234 # hard-coded ._.

    with st.sidebar:
        st.subheader("Settings")

        # source of evidence sources
        evidence_type = st.radio(
            label="Evidence sources",
            options=["Gold evidence", "Custom retrieved evidence"],
            index=0,
        )
        dataset = "averitec" if evidence_type == "Gold evidence" else "averitec_web_evidence"
        cc_metrics_results_path = f"results/results_{dataset}_longcite.json"
        
        # attribution method
        attr_method_name = st.radio(
            label="Attribution method",
            options=["Contributive attribution (ContextCite)", 
                     "Direct attribution (LongCite-8B)", 
                     "Post-hoc attribution (NLI: DeBERTa-v3)",
                     "Post-hoc attribution (Semantic similarity)"],
            index=0,
        )
        eval_metrics_results_path = f"results_final/{dataset}/LongCite-llama3.1-8b_{LABEL2METH[attr_method_name]}_{dataset}_deepseek-chat.json"

        # claim index
        pos = st.number_input(
            label="Claim",
            value=1,
            min_value=1,
            max_value=n_claims,
            help="Claim of which the results should be displayed",
            placeholder="Claim Index",
            step=1,
            format="%d"
        ) - 1

        order_with_featured_examples = get_order_with_featured_examples(n_claims, FEATURED_EXAMPLE_IDXS)
        idx = order_with_featured_examples[pos]

    return cc_metrics_results_path, eval_metrics_results_path, idx, attr_method_name

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

def render_ground_truth_justification(eval_metrics_results, col_l):

    gt_justification = eval_metrics_results["justification"]
    with col_l:
        st.markdown(f"""
                    <div>
                        <span class="section-label">Ground truth justification:</span> {gt_justification}
                    </div>
        """, unsafe_allow_html=True) 

def render_model_answer(results, span2citestr, col_l):

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
            answer_text += f"<span style='background-color: {color};'>{process_text(statement)}</span>"
            answer_text += " "

            # apply color to the citation depending on whether it is relevant or not
            cite_colors = [RELEVANT_SCORE2COLOR[relevant_score] for relevant_score in relevant_scores]
            for cite_str, col in zip(sent_citation_strs, cite_colors):
                answer_text += f"<span style='color: {col};'>{span2citestr.get(cite_str)}</span>"

            answer_text += "\n\n"

        return answer_text
    

    answer_text = build_answer_text(results)
    with col_l:
        st.markdown(f"""
            <div class="justification" id="justification">
                <span class="section-label">Justification:</span> 
                {answer_text}
        """, unsafe_allow_html=True)  # why on earth?

def render_evidences(eval_metrics_results, cc_metrics_results, longcite_cc_metrics_results, attr_method, span2citestr, col_r):

    # def build_log_prob_drops_table(drops: dict):
    #     headers = "".join(f'<th>{k}</th>' for k in drops.keys())
    #     values = "".join(f'<td>{v:.2f}</td>' for v in drops.values())

    #     return f"""
    #         <table style="border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; width: auto;">
    #             <thead>
    #                 <tr>
    #                     <th>k</th>
    #                     {headers}
    #                 </tr>
    #             </thead>
    #             <tbody>
    #                 <tr>
    #                     <td class="section-label">Top-k log-prob drop</td>
    #                     {values}
    #                 </tr>
    #             </tbody>
    #         </table>
    #     """

    with col_r:
        st.markdown(f"""
            <div class="evidence-heading">
                Evidence snippets:
            </div>
        """, unsafe_allow_html=True) 

    # helpers for URL text fragments (highlighting the cited parts of the websites)
    def enc(string): 
        return quote(string, safe="").replace("-", "%2D")
    
    def text_fragment_url(url, text):
        return f"{url}#:~:text={enc(text)}"
    

    longcite_citations = longcite_cc_metrics_results["citations"]
    log_prob_drops_dict = cc_metrics_results["metrics"]["top_k_drop"]
    statements = eval_metrics_results["statements"]
    citations = cc_metrics_results["citations"]

    # build evidence html
    evidence_html = ""
    for i in range(len(statements)):
        sc = statements[i]
        citation = sc["citation"]

        # longcite_citations_sent = longcite_citations[i]
        # if longcite_citations_sent:  
        #     k_longcite = len(longcite_citations_sent)

        # if attr_method == "longcite_llm_direct":
        #     drops = {
        #         f"k={k_longcite}": log_prob_drops_dict["top_k_drop_longcite"][j],
        #     }
        # else: 
        #     drops = {
        #         "k=1": log_prob_drops_dict["top_1_drop"][j],
        #         "k=3": log_prob_drops_dict["top_3_drop"][j],
        #         "k=5": log_prob_drops_dict["top_5_drop"][j],
        #         f"k={k_longcite}": log_prob_drops_dict["top_k_drop_longcite"][j],
        #     }

        # k_self = len(citations[j])
        # if log_prob_drops_dict["top_k_drop_citations"][j] is not None:
        #     drops = {
        #         f"k={k_self}": log_prob_drops_dict["top_k_drop_citations"][j],
        #     }
        # else:
        #     drops = {}
        # j += 1

        if citation:
            evidence_html += f"""
                <div class="statement-header">
                    Statement {i+1}:
                </div>
            """    

            for c in citation:
                cite_text, cite_span, cite_score, relevant_score, source_url = process_text(c["cite"]), c["span"], c["score"], c["relevant_score"], c["source_url"]
                start, end = cite_span 
                if start == end:
                    cite_str = f"[{start}]"
                else:
                    cite_str = f"[{start}-{end}]"

                # if cite_score:        
                #     score_html = f"<span>Score: {cite_score:.2f}</span>"
                # else:
                #     score_html = f"<span>Score: -</span>"

                relevant_text = "Relevant" if relevant_score == 1 else "Irrelevant"
                relevant_color = RELEVANT_SCORE2COLOR[relevant_score]

                # text fragment URL
                tf_url = text_fragment_url(source_url, c["cite"])

                evidence_html += f"""
                <div class="evidence-card">
                    <div class="evidence-title">
                        <span style='color: {relevant_color};'>{relevant_text}</span>
                    </div>
                    <div class="evidence-title">
                        <span>Evidence snippet {span2citestr.get(cite_str)}</span>
                    </div>
                    <div class="evidence-content">{cite_text}</div>
                    <div class="source-url">
                        <a href='{tf_url}' target='_blank'>{source_url}</a>
                    </div>
                </div>
            """

            # # log-prob drops table
            # if citation:
            #     table_html = build_log_prob_drops_table(drops)
            #     evidence_html += f"<div class='drops-table'>{table_html}</div>"

            ## display citation precision, recall, and faithfulness
            # get scores
            relevant_scores = [c["relevant_score"] for c in citation]
            n_relevant_citations = len([score for score in relevant_scores if score == 1])
            citation_precision = n_relevant_citations/len(relevant_scores)
            citation_recall = sc["support_score"]
            log_prob_drop = log_prob_drops_dict["top_k_drop_citations"][i]
            prob_ratio = math.exp(log_prob_drop)

            # build explaining strings
            precision_str = f"{n_relevant_citations}/{len(relevant_scores)}"
            recall_str = "Fully supported" if citation_recall == 1 else "Partially supported" if citation_recall == 0.5 else "Not supported"
            faithfulness_str = f"~{prob_ratio:.1f}x less likely"

            # colors
            if citation_precision <= 0.1:
                precision_color = UNSUPPORTED_COLOR
            elif citation_precision >= 0.9:
                precision_color = SUPPORTED_COLOR
            else:
                precision_color = PARTIAL_SUPPORT_COLOR 
            recall_color = SUPPORT_SCORE2COLOR[citation_recall]
            if log_prob_drop <= 0.5:
                faithfulness_color = UNSUPPORTED_COLOR
            elif log_prob_drop >= 1.5:
                faithfulness_color = SUPPORTED_COLOR
            else:
                faithfulness_color = PARTIAL_SUPPORT_COLOR

            metrics_html = f"""
            <div class="citation-metrics">
                <div class="citation-metrics-header">Statement {i+1} Metrics</div>
                <div class="metrics-grid">
                    <div class="metric-row" style="--metric-color: {precision_color};">
                        <span class="metric-label">Citation precision</span>
                        <span class="metric-value">{citation_precision:.2g}</span>
                        <span class="metric-desc">{precision_str} relevant</span>
                    </div>
                    <div class="metric-row" style="--metric-color: {recall_color};">
                        <span class="metric-label">Citation recall</span>
                        <span class="metric-value">{citation_recall:g}</span>
                        <span class="metric-desc">{recall_str}</span>
                    </div>
                    <div class="metric-row" style="--metric-color: {faithfulness_color};">
                        <span class="metric-label">Citation faithfulness</span>
                        <span class="metric-value">{log_prob_drop:.2f}</span>
                        <span class="metric-desc">{faithfulness_str}</span>
                    </div>
                </div>
            </div>
            """
            evidence_html += metrics_html

    with col_r:
        st.html(f"<div class='evidence-container'>{evidence_html}</div>")
            
def main():
    st.set_page_config(layout="wide")
    title_col, help_col = st.columns([5,1], vertical_alignment="center")
    with title_col:
        st.title("Faithfulness and correctness of RAG attributions")

    with help_col:
        if st.button("How to read this page", icon=":material/info:", use_container_width=False, type="primary"):
            show_explanations()

    # get settings
    cc_metrics_results_path, eval_metrics_results_path, idx, attr_method_name = render_sidebar()
    eval_metrics_results = load_json_from_hub(eval_metrics_results_path)[idx]
    cc_metrics_results_all_methods = load_json_from_hub(cc_metrics_results_path)["results"][idx]

    attr_method = get_attr_method(str(eval_metrics_results_path))
    st.subheader(f"Attribution method: {attr_method_name}")
    cc_metrics_results = cc_metrics_results_all_methods["methods"][attr_method]
    longcite_cc_metrics_results = cc_metrics_results_all_methods["methods"]["longcite_llm_direct"]  # assuming LongCite is always used

    # mapping to simplify the cite_strs to start at 1,2,3,...
    span2citestr = get_mapping_spans2cite_strs(eval_metrics_results)

    # layout
    col_l, col_r = st.columns([1,1])

    # display content
    render_claim(eval_metrics_results, col_l)
    render_verdicts(eval_metrics_results, col_l)
    # render_ground_truth_justification(eval_metrics_results, col_l)
    render_model_answer(eval_metrics_results, span2citestr, col_l)
    render_evidences(eval_metrics_results, cc_metrics_results, longcite_cc_metrics_results, attr_method, span2citestr, col_r)
    

if __name__ == "__main__":
    main()