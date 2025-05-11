# tabs/ragas_eval_tab.py
import streamlit as st
import os
import json
import pandas as pd


# RAGAS and Langchain specific imports will be conditional
# We'll pass RAGAS_AVAILABLE status and clients from the main app

# --- RAGAS Evaluation Logic ---
def get_ragas_clients_conditional(_app_config, RAGAS_AVAILABLE_FLAG):
    if not RAGAS_AVAILABLE_FLAG:
        return None, None

    # These imports are only needed if RAGAS is available
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    required_keys_llm = ["AZURE_OPENAI_VERSION", "AZURE_OPENAI_BASE", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    required_keys_emb = ["AZURE_API_VERSION_EMBEDDING", "AZURE_OPENAI_BASE", "AZURE_API_KEY_EMBEDDING",
                         "AZURE_ENGINE_EMBEDDING"]

    if not all(key in _app_config for key in required_keys_llm) or \
            not all(key in _app_config for key in required_keys_emb):
        st.error("Missing some Azure OpenAI or Embedding configurations in .env for RAGAS.")
        return None, None

    try:
        llm_client = AzureChatOpenAI(
            openai_api_version=_app_config["AZURE_OPENAI_VERSION"],
            azure_endpoint=_app_config["AZURE_OPENAI_BASE"],
            api_key=_app_config["AZURE_OPENAI_KEY"],
            azure_deployment=_app_config["AZURE_OPENAI_DEPLOYMENT"],
            temperature=0
        )
        azure_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=_app_config["AZURE_OPENAI_BASE"],
            api_key=_app_config["AZURE_API_KEY_EMBEDDING"],
            api_version=_app_config["AZURE_API_VERSION_EMBEDDING"],
            azure_deployment=_app_config.get("AZURE_ENGINE_EMBEDDING", "text-embedding-3-large"),
            model=_app_config.get("AZURE_ENGINE_EMBEDDING")
        )
        ragas_llm = LangchainLLMWrapper(llm_client)
        ragas_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)
        return ragas_llm, ragas_embeddings
    except Exception as e:
        st.error(f"Error initializing RAGAS clients: {e}")
        return None, None


def run_ragas_evaluation_st(uploaded_files, ragas_output_filename, ragas_llm, ragas_embeddings, RAGAS_AVAILABLE_FLAG):
    if not RAGAS_AVAILABLE_FLAG:
        st.error("RAGAS libraries are not available. Cannot run evaluation.")
        return None
    if not ragas_llm or not ragas_embeddings:
        st.error("RAGAS LLM or Embeddings clients not initialized.")
        return None

    # Conditional imports for RAGAS execution
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall

    evaluation_results_for_json = []
    run_config = RunConfig(timeout=180, max_workers=4)

    metrics_to_run = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm)
    ]

    output_dir = "ragas_eval_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not ragas_output_filename:  # Default filename if not provided
        ragas_output_filename = os.path.join(output_dir, "ragas_evaluation_summary.json")
    elif not os.path.dirname(ragas_output_filename):  # If only filename is given, put it in default_dir
        ragas_output_filename = os.path.join(output_dir, os.path.basename(ragas_output_filename))
    # Else, use the full path provided by the user.

    if os.path.exists(ragas_output_filename):
        try:
            os.remove(ragas_output_filename)
            st.info(f"Removed existing RAGAS output file: {ragas_output_filename}")
        except OSError as e:
            st.warning(f"Could not remove existing RAGAS output file {ragas_output_filename}: {e}")

    progress_bar_files = st.progress(0)
    status_text_files = st.empty()

    for file_idx, uploaded_file in enumerate(uploaded_files):
        status_text_files.info(f"Processing file: {uploaded_file.name} ({file_idx + 1}/{len(uploaded_files)})")
        try:
            # Reset file pointer for json.load, in case it was read before
            uploaded_file.seek(0)
            data = json.load(uploaded_file)
            if not isinstance(data, list):
                st.warning(f"File {uploaded_file.name} is not a list of records. Skipping.")
                continue
        except Exception as e:
            st.error(f"Error reading JSON from {uploaded_file.name}: {e}")
            continue

        all_questions, all_generated_answers, all_contexts, all_ground_truths = [], [], [], []

        for doc_idx, doc in enumerate(data):
            question = doc.get('question')
            contexts = doc.get('retrieved_context')
            ref_answer = doc.get('ref_answer')
            generated_answer = doc.get('generated_answer')

            if not all([question, contexts, ref_answer, generated_answer]):
                st.warning(
                    f"Skipping record {doc_idx + 1} in {uploaded_file.name} due to missing fields (question, retrieved_context, ref_answer, generated_answer).")
                continue
            if not isinstance(contexts, list):
                st.warning(
                    f"Record {doc_idx + 1} in {uploaded_file.name} 'retrieved_context' is not a list. Wrapping it. Context: {contexts}")
                contexts = [str(contexts)]  # Ensure it's a list of strings

            all_questions.append(question)
            all_generated_answers.append(generated_answer)
            all_contexts.append(contexts)
            all_ground_truths.append(ref_answer)

        if not all_questions:
            st.warning(f"No valid data found in {uploaded_file.name} to evaluate. Skipping.")
            progress_bar_files.progress((file_idx + 1) / len(uploaded_files))
            continue

        dataset_dict = {
            "question": all_questions,
            "answer": all_generated_answers,
            "contexts": all_contexts,
            "ground_truth": all_ground_truths
        }
        dataset = Dataset.from_dict(dataset_dict)

        st.write(f"Evaluating {len(all_questions)} items from {uploaded_file.name} with RAGAS...")
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics_to_run,
            run_config=run_config
        )

        file_summary = {
            "path": uploaded_file.name,
            "total_questions": len(all_questions),
        }

        result_df = eval_result.to_pandas()  # Convert dataset with scores to DataFrame
        for metric in metrics_to_run:
            metric_name_underscored = metric.name.replace(' ', '_').lower()
            # RAGAS evaluate returns a dictionary of average scores
            file_summary[f"average_score_{metric_name_underscored}"] = eval_result.get(metric.name, None)
            # And the per-item scores are columns in the returned dataset (now in result_df)
            if metric.name in result_df.columns:
                file_summary[f"total_scores_{metric_name_underscored}"] = result_df[metric.name].tolist()
            else:
                file_summary[f"total_scores_{metric_name_underscored}"] = []

        evaluation_results_for_json.append(file_summary)
        st.success(f"Finished RAGAS evaluation for {uploaded_file.name}.")
        df_display = pd.DataFrame([file_summary])
        st.dataframe(df_display)

        progress_bar_files.progress((file_idx + 1) / len(uploaded_files))

    status_text_files.success("All files processed by RAGAS.")

    if evaluation_results_for_json:
        with open(ragas_output_filename, "w") as f:
            json.dump(evaluation_results_for_json, f, indent=4)
        st.success(f"RAGAS evaluation summary saved to: {ragas_output_filename}")
        return ragas_output_filename
    else:
        st.warning("No evaluation results were generated.")
        return None


def show_ragas_eval_tab(app_config, RAGAS_AVAILABLE_FLAG):
    """Displays the RAGAS Evaluation Tab."""
    st.header("⚖️ RAGAS Evaluation")

    if not RAGAS_AVAILABLE_FLAG:
        st.error(
            "RAGAS libraries are not installed or importable. This tab is disabled. "
            "Please install them: `pip install ragas langchain-openai datasets`"
        )
        return
    if app_config is None:
        st.error("Application configuration (.env) not loaded. RAGAS evaluation cannot proceed.")
        return

    st.markdown("""
    Upload one or more JSON files containing RAG outputs. Each file should be a list of objects,
    and each object must contain:
    - `question`: The input question.
    - `generated_answer`: The answer generated by the RAG system.
    - `retrieved_context`: A list of context strings used for generation.
    - `ref_answer`: The ground truth reference answer.
    """)
    uploaded_rag_outputs = st.file_uploader(
        "Upload RAG output JSON files for RAGAS evaluation",
        type=['json'],
        accept_multiple_files=True,
        key="ragas_input_uploader_tab"  # Changed key
    )
    ragas_eval_output_filename = st.text_input(
        "Output file name for RAGAS evaluation results (e.g., ragas_summary.json):",
        placeholder="my_ragas_evaluation.json",
        key="ragas_output_name_tab"  # Changed key
    ).strip()

    if st.button("⚖️ Run RAGAS Evaluation", type="primary", use_container_width=True,
                 key="run_ragas_eval_tab"):  # Changed key
        if not uploaded_rag_outputs:
            st.warning("Please upload at least one RAG output JSON file.")
        else:
            with st.spinner("Initializing RAGAS clients and running evaluation... This may take a while."):
                # Get RAGAS clients conditionally
                ragas_llm_client, ragas_embedding_client = get_ragas_clients_conditional(app_config,
                                                                                         RAGAS_AVAILABLE_FLAG)

                if ragas_llm_client and ragas_embedding_client:
                    st.info(
                        f"Using RAGAS LLM: {ragas_llm_client.llm.model_name if hasattr(ragas_llm_client, 'llm') and hasattr(ragas_llm_client.llm, 'model_name') else 'Unknown'}")
                    st.info(
                        f"Using RAGAS Embeddings: {ragas_embedding_client.embeddings.model if hasattr(ragas_embedding_client, 'embeddings') and hasattr(ragas_embedding_client.embeddings, 'model') else 'Unknown'}")

                    final_ragas_output_path = run_ragas_evaluation_st(
                        uploaded_rag_outputs,
                        ragas_eval_output_filename,
                        ragas_llm_client,
                        ragas_embedding_client,
                        RAGAS_AVAILABLE_FLAG
                    )
                    if final_ragas_output_path and os.path.exists(final_ragas_output_path):
                        st.success(f"RAGAS Evaluation complete. Results saved to: {final_ragas_output_path}")
                        try:
                            with open(final_ragas_output_path, "rb") as fp_ragas:
                                st.download_button(
                                    label="Download RAGAS Evaluation JSON",
                                    data=fp_ragas,
                                    file_name=os.path.basename(final_ragas_output_path),
                                    mime="application/json",
                                    key="download_ragas_json_tab"  # Changed key
                                )
                        except Exception as e:
                            st.error(f"Error offering download for RAGAS output: {e}")
                    else:
                        st.error("RAGAS evaluation did not produce an output file or an error occurred.")
                else:
                    st.error("Failed to initialize RAGAS clients. Check .env configuration and console logs.")
