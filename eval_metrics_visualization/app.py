import streamlit as st 
from pathlib import Path 

def render_sidebar():
    app_path = Path(__file__).resolve()
    cc_metrics_results_dir = app_path.parent.parent / "context_cite_metrics/results"
    eval_metrics_results_dir = app_path.parent.parent / "context_cite_metrics/results_final/AVeriTeC/old_prompt"

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
            label="ContextCite metrics results file",
            options=[path.name for path in eval_metrics_results_dir.iterdir() if path.is_file()],
            index=None,
            placeholder="ContextCite metrics results file", 
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

def main():
    st.title("Evaluation metrics visualization")

    # get settings
    cc_metrics_results_path, eval_metrics_results_path, idx = render_sidebar()

    


if __name__ == "__main__":
    main()