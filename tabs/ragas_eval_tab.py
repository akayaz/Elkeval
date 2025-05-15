# tabs/ragas_eval_tab.py
import streamlit as st
import os
import json
import pandas as pd
import db_utils  # Uses ES now
from datetime import datetime


# Conditional RAGAS imports remain the same
def get_ragas_clients_conditional(_app_config, RAGAS_AVAILABLE_FLAG):
    if not RAGAS_AVAILABLE_FLAG: return None, None
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    required_keys_llm = ["AZURE_OPENAI_VERSION", "AZURE_OPENAI_BASE", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    required_keys_emb = ["AZURE_API_VERSION_EMBEDDING", "AZURE_OPENAI_BASE", "AZURE_API_KEY_EMBEDDING",
                         "AZURE_ENGINE_EMBEDDING"]
    if not all(key in _app_config for key in required_keys_llm) or not all(
            key in _app_config for key in required_keys_emb):
        st.error("Missing Azure OpenAI/Embedding configurations in .env for RAGAS.");
        return None, None
    try:
        llm_client = AzureChatOpenAI(openai_api_version=_app_config["AZURE_OPENAI_VERSION"],
                                     azure_endpoint=_app_config["AZURE_OPENAI_BASE"],
                                     api_key=_app_config["AZURE_OPENAI_KEY"],
                                     azure_deployment=_app_config["AZURE_OPENAI_DEPLOYMENT"], temperature=0)
        azure_embeddings = AzureOpenAIEmbeddings(azure_endpoint=_app_config["AZURE_OPENAI_BASE"],
                                                 api_key=_app_config["AZURE_API_KEY_EMBEDDING"],
                                                 #api_version=_app_config["AZURE_API_VERSION_EMBEDDING"],
                                                 azure_deployment=_app_config.get("AZURE_ENGINE_EMBEDDING",
                                                                                  "text-embedding-3-large"),
                                                 model=_app_config.get("AZURE_EMBEDDING_MODEL",
                                                                       "text-embedding-ada-002"))
        ragas_llm = LangchainLLMWrapper(llm_client);
        ragas_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)
        return ragas_llm, ragas_embeddings
    except Exception as e:
        st.error(f"Error initializing RAGAS clients: {e}"); return None, None


def run_ragas_evaluation_st(uploaded_files, ragas_output_filename_base, ragas_llm, ragas_embeddings,
                            RAGAS_AVAILABLE_FLAG):
    if not RAGAS_AVAILABLE_FLAG: st.error("RAGAS libraries not available."); return None, None, [], []
    if not ragas_llm or not ragas_embeddings: st.error(
        "RAGAS LLM or Embeddings clients not initialized."); return None, None, [], []

    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall

    all_evaluation_summaries_for_json_file = []  # This list will be saved to the JSON file

    run_config = RunConfig(timeout=360, max_workers=4)
    metrics_to_run = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm)
    ]
    output_dir = "ragas_eval_outputs";
    os.makedirs(output_dir, exist_ok=True)

    if not ragas_output_filename_base: ragas_output_filename_base = "ragas_evaluation_results"
    if not ragas_output_filename_base.endswith(".json"): ragas_output_filename_base += ".json"
    final_ragas_summary_filepath = os.path.join(output_dir, ragas_output_filename_base)

    if os.path.exists(final_ragas_summary_filepath):
        try:
            os.remove(final_ragas_summary_filepath); st.info(
                f"Removed existing RAGAS output file: {final_ragas_summary_filepath}")
        except OSError as e:
            st.warning(f"Could not remove {final_ragas_summary_filepath}: {e}")

    progress_bar_files = st.progress(0, text="Initializing RAGAS Evaluation...")
    status_text_files = st.empty()
    status_text_files.text("Preparing to evaluate uploaded files...")

    total_files = len(uploaded_files)
    processed_input_filenames_for_experiment = []  # For saving to experiment DB
    all_avg_metrics_for_experiment = {}  # To store a single set of avg metrics if user saves experiment
    all_retrieval_methods_for_experiment = set()

    for file_idx, uploaded_file in enumerate(uploaded_files):
        current_file_name = uploaded_file.name
        processed_input_filenames_for_experiment.append(current_file_name)  # Add to list for experiment
        status_text_files.info(f"({file_idx + 1}/{total_files}) Loading data from: {current_file_name}...")

        master_questions, master_generated_answers, master_contexts, master_ground_truths = [], [], [], []

        try:
            uploaded_file.seek(0);
            data = json.load(uploaded_file)
            if not isinstance(data, list):
                st.warning(f"File {current_file_name} is not a list of records. Skipping this file.");
                progress_bar_files.progress((file_idx + 1) / total_files)
                continue
        except Exception as e:
            st.error(f"Error reading JSON from {current_file_name}: {e}. Skipping this file.")
            progress_bar_files.progress((file_idx + 1) / total_files)
            continue

        filename_parts = os.path.basename(current_file_name).split('_')
        if len(filename_parts) > 1: all_retrieval_methods_for_experiment.add(filename_parts[0])

        for doc_idx, doc in enumerate(data):
            question = doc.get('question');
            contexts = doc.get('raw_retrieved_contexts_from_es')
            ref_answer = doc.get('ref_answer');
            generated_answer = doc.get('generated_answer')
            if not all([question, contexts, ref_answer, generated_answer]):
                st.warning(f"Skipping record {doc_idx + 1} in {current_file_name} due to missing fields.");
                continue
            if not isinstance(contexts, list): contexts = [str(contexts)]
            master_questions.append(question);
            master_generated_answers.append(generated_answer)
            master_contexts.append(contexts);
            master_ground_truths.append(ref_answer)

        if not master_questions:
            st.warning(f"No valid data found in {current_file_name} to evaluate. Skipping this file.")
            progress_bar_files.progress((file_idx + 1) / total_files)
            continue

        dataset_dict = {"question": master_questions, "answer": master_generated_answers, "contexts": master_contexts,
                        "ground_truth": master_ground_truths}
        dataset = Dataset.from_dict(dataset_dict)

        status_text_files.info(
            f"({file_idx + 1}/{total_files}) Evaluating {len(master_questions)} items from {current_file_name} with RAGAS...")
        eval_result_dataset = None
        evaluation_succeeded = False
        with st.spinner(f"RAGAS is evaluating {current_file_name}... This may take a while."):
            try:
                eval_result_dataset = evaluate(dataset=dataset, metrics=metrics_to_run, run_config=run_config)
                evaluation_succeeded = True
            except Exception as e:
                st.error(f"An error occurred during RAGAS evaluation for file {current_file_name}: {e}")

        file_summary_for_json = {"path": current_file_name, "total_questions": len(master_questions)}
        per_item_scores_for_file = {}
        avg_scores_for_file = {}

        if evaluation_succeeded and eval_result_dataset:
            status_text_files.info(f"({file_idx + 1}/{total_files}) Processing results for {current_file_name}...")
            try:
                result_df = eval_result_dataset.to_pandas()
                for metric in metrics_to_run:
                    metric_name_key = metric.name.replace(' ', '_').lower()
                    if metric.name in result_df.columns:
                        avg_scores_for_file[f"avg_{metric_name_key}"] = result_df[metric.name].mean()
                        per_item_scores_for_file[metric.name] = result_df[metric.name].tolist()
                    else:
                        avg_scores_for_file[f"avg_{metric_name_key}"] = None
                        per_item_scores_for_file[metric.name] = []
                        st.warning(f"Metric '{metric.name}' not found in RAGAS results for {current_file_name}.")
            except Exception as e:
                st.error(f"Error processing RAGAS results for {current_file_name}: {e}")
        elif evaluation_succeeded and not eval_result_dataset:
            st.warning(f"RAGAS evaluation completed for {current_file_name} but returned no result object.")

        file_summary_for_json["average_scores"] = avg_scores_for_file
        file_summary_for_json["per_item_scores"] = per_item_scores_for_file
        all_evaluation_summaries_for_json_file.append(
            file_summary_for_json)  # Append individual file summary to list for JSON

        # For overall experiment metrics, we might average the averages if multiple files are one experiment
        # For simplicity now, if saving an experiment, we'll use the averages from the *last processed file*
        # or an average of averages. Let's average the averages.
        for key, val in avg_scores_for_file.items():
            if val is not None:
                all_avg_metrics_for_experiment.setdefault(key, []).append(val)

        if evaluation_succeeded:
            st.success(f"Finished RAGAS evaluation for {current_file_name}.")
        else:
            st.warning(f"RAGAS evaluation for {current_file_name} did not complete successfully.")

        df_display = pd.DataFrame([{"File": file_summary_for_json["path"],
                                    "Questions": file_summary_for_json["total_questions"],
                                    "Avg Faithfulness": f"{avg_scores_for_file.get('avg_faithfulness', 0) * 100:.1f}%",
                                    "Avg Answer Relevancy": f"{avg_scores_for_file.get('avg_answer_relevancy', 0) * 100:.1f}%"
                                    }])
        st.dataframe(df_display)
        progress_bar_files.progress((file_idx + 1) / total_files, text=f"Completed {file_idx + 1}/{total_files} files.")

    status_text_files.success("All RAGAS evaluations complete.")

    # Calculate final average metrics for the experiment (average of averages if multiple files)
    final_experiment_avg_metrics = {}
    for key, values_list in all_avg_metrics_for_experiment.items():
        if values_list:
            final_experiment_avg_metrics[key.replace('avg_', '')] = sum(values_list) / len(
                values_list)  # Store without 'avg_' prefix for DB
        else:
            final_experiment_avg_metrics[key.replace('avg_', '')] = None

    if all_evaluation_summaries_for_json_file:
        with open(final_ragas_summary_filepath, "w") as f:
            json.dump(all_evaluation_summaries_for_json_file, f, indent=4)
        st.success(f"RAGAS evaluation results for all files saved to: {final_ragas_summary_filepath}")
        # Return data needed for potential experiment saving
        return final_ragas_summary_filepath, final_experiment_avg_metrics, processed_input_filenames_for_experiment, list(
            all_retrieval_methods_for_experiment)
    else:
        st.warning("No evaluation results were generated for any file.")
        return None, None, [], []


def show_ragas_eval_tab(app_config, RAGAS_AVAILABLE_FLAG, es_client):  # Added es_client
    st.header("⚖️ RAGAS Evaluation")
    if not RAGAS_AVAILABLE_FLAG: st.error("RAGAS libraries not installed."); return
    if app_config is None: st.error("App config not loaded."); return
    # No explicit check for es_client here, as saving is optional. db_utils will handle it.

    st.markdown("Upload RAG output JSON files. Each file will be evaluated, and results compiled.")
    uploaded_rag_outputs = st.file_uploader("Upload RAG output JSON files", type=['json'], accept_multiple_files=True,
                                            key="ragas_input_uploader_tab")
    ragas_eval_output_filename_base = st.text_input("Output filename for compiled RAGAS results:",
                                                    placeholder="compiled_ragas_results",
                                                    key="ragas_output_name_tab").strip()

    if 'last_ragas_run_details_for_exp' not in st.session_state:  # Renamed for clarity
        st.session_state.last_ragas_run_details_for_exp = None

    if st.button("⚖️ Run RAGAS Evaluation", type="primary", use_container_width=True, key="run_ragas_eval_tab"):
        st.session_state.last_ragas_run_details_for_exp = None
        if not uploaded_rag_outputs:
            st.warning("Please upload RAG output JSON file(s).")
        else:
            ragas_llm_client, ragas_embedding_client = get_ragas_clients_conditional(app_config, RAGAS_AVAILABLE_FLAG)
            if ragas_llm_client and ragas_embedding_client:
                results = run_ragas_evaluation_st(
                    uploaded_rag_outputs, ragas_eval_output_filename_base,
                    ragas_llm_client, ragas_embedding_client, RAGAS_AVAILABLE_FLAG
                )
                if results and len(results) == 4:
                    final_ragas_output_path, avg_metrics_for_exp, processed_inputs, inferred_methods = results
                    if final_ragas_output_path and os.path.exists(final_ragas_output_path):
                        st.session_state.last_ragas_run_details_for_exp = {
                            "summary_file_path": final_ragas_output_path,
                            "avg_metrics": avg_metrics_for_exp,  # This is now the overall average for the experiment
                            "processed_input_filenames": processed_inputs,
                            "inferred_methods": inferred_methods
                        }
                        try:
                            with open(final_ragas_output_path, "rb") as fp_ragas:
                                st.download_button("Download Compiled RAGAS Results JSON", fp_ragas,
                                                   os.path.basename(final_ragas_output_path), "application/json",
                                                   key="download_ragas_json_tab")
                        except Exception as e:
                            st.error(f"Error offering download: {e}")
                    # elif final_ragas_output_path: # Error message handled in run_ragas_evaluation_st
                    #         st.error(f"RAGAS evaluation ran but output file not found at {final_ragas_output_path}.")
                # else:  # Error message handled in run_ragas_evaluation_st
                #    st.error("RAGAS evaluation function did not return the expected results. Check logs.")
            else:
                st.error("Failed to initialize RAGAS clients.")

    if st.session_state.last_ragas_run_details_for_exp:
        st.divider();
        st.subheader("💾 Save as Experiment")
        run_details = st.session_state.last_ragas_run_details_for_exp

        default_exp_name = f"Exp_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if run_details.get(
            'inferred_methods'): default_exp_name += f"_{'_'.join(sorted(list(set(run_details['inferred_methods']))))}"
        if run_details.get('processed_input_filenames') and run_details['processed_input_filenames']:
            first_input_base = os.path.splitext(os.path.basename(run_details['processed_input_filenames'][0]))[0]
            for method_prefix in run_details.get('inferred_methods', []):
                if first_input_base.startswith(method_prefix + "_"): first_input_base = first_input_base[
                                                                                        len(method_prefix) + 1:]; break
            default_exp_name += f"_{first_input_base[:15]}"
        exp_name = st.text_input("Experiment Name:", value=default_exp_name, key="exp_name_input")

        if st.button("Save to Experiments Dashboard", key="save_exp_button_es"):
            if not exp_name.strip():
                st.warning("Please provide an experiment name.")
            elif es_client is None:
                st.error("Elasticsearch client not available. Cannot save experiment.")
            else:
                # avg_metrics in run_details already has 'avg_' prefix removed by run_ragas_evaluation_st
                metrics_to_save = run_details.get('avg_metrics') if isinstance(run_details.get('avg_metrics'),
                                                                               dict) else {}

                saved_id = db_utils.save_experiment(
                    es_client=es_client, name=exp_name.strip(),
                    ragas_input_files=run_details.get('processed_input_filenames', []),
                    ragas_output_file=run_details.get('summary_file_path', ""),
                    metrics=metrics_to_save,  # Should not have 'avg_' prefix here for DB schema
                    retrieval_methods=run_details.get('inferred_methods', [])
                )
                if saved_id:  # save_experiment now returns ID or None
                    # st.success(f"Experiment '{exp_name.strip()}' saved successfully!") # Message handled by db_utils
                    st.session_state.last_ragas_run_details_for_exp = None
