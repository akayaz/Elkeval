# tabs/rag_processor_tab.py
import streamlit as st
import os
import json
import pandas as pd
import retriever_utils as utils
import traceback

# --- Configuration for retrieval methods ---
# 'lexical_field_required': True if the method's lexical part uses field_to_query
# 'semantic_field_required': True if the method's semantic part uses semantic_text_field
METHOD_INFO = {
    'Full-Text Search': {'func': utils.query_fulltext, 'type': 'fulltext', 'lexical_field_required': True,
                         'semantic_field_required': False},
    'Semantic Search': {'func': utils.query_semantic, 'type': 'semantic', 'lexical_field_required': False,
                        'semantic_field_required': True},
    'Hybrid Search (RRF)': {'func': utils.query_hybrid, 'type': 'hybrid', 'lexical_field_required': True,
                            'semantic_field_required': True},
    'Hybrid Search (Linear Combination)': {'func': utils.query_hybrid_linear, 'type': 'hybrid',
                                           'lexical_field_required': True, 'semantic_field_required': True},
    'BGE Semantic Search': {'func': utils.query_semantic_bge, 'type': 'semantic', 'lexical_field_required': False,
                            'semantic_field_required': True, 'vector_field_prefix': 'content_semantic_bge-m3'},
    # vector_field_prefix is a hint for UI default
    'BGE Hybrid Search (RRF)': {'func': utils.query_hybrid_bge, 'type': 'hybrid', 'lexical_field_required': True,
                                'semantic_field_required': True, 'vector_field_prefix': 'content_semantic_bge-m3'},
    'BGE Hybrid Search (Linear)': {'func': utils.query_hybrid_linear_bge, 'type': 'hybrid',
                                   'lexical_field_required': True, 'semantic_field_required': True,
                                   'vector_field_prefix': 'content_semantic_bge-m3'},
    'Hybrid Rerank (Cohere)': {'func': utils.query_hybrid_rerank, 'type': 'hybrid_rerank',
                               'lexical_field_required': True, 'semantic_field_required': True},
    'BGE Hybrid Rerank (Cohere)': {'func': utils.query_hybrid_bge_rerank, 'type': 'hybrid_rerank',
                                   'lexical_field_required': True, 'semantic_field_required': True,
                                   'vector_field_prefix': 'content_semantic_bge-m3'}
}


class QuestionProcessor:
    def __init__(self, es_client, openai_client, openai_model_name, method_name, index_name,
                 field_to_query,
                 semantic_text_field,
                 primary_content_field_for_llm, rerank_content_field=None):
        self.es_client = es_client
        self.openai_client = openai_client
        self.openai_model_name = openai_model_name
        self.method_name = method_name
        self.index_name = index_name
        self.field_to_query = field_to_query
        self.semantic_text_field = semantic_text_field
        self.primary_content_field_for_llm = primary_content_field_for_llm
        self.rerank_content_field = rerank_content_field if rerank_content_field else primary_content_field_for_llm

        if self.method_name not in METHOD_INFO:
            raise ValueError(f"Unknown method '{self.method_name}' selected.")
        self.query_function_ptr = METHOD_INFO[self.method_name]['func']
        self.method_type = METHOD_INFO[self.method_name]['type']
        self.lexical_field_required = METHOD_INFO[self.method_name]['lexical_field_required']
        self.semantic_field_required = METHOD_INFO[self.method_name]['semantic_field_required']

    def process_single_question(self, question_text):
        elasticsearch_results = []
        try:
            query_args = {
                "es_client": self.es_client, "query": question_text, "index_name": self.index_name
            }
            if self.lexical_field_required: query_args["field_to_query"] = self.field_to_query
            if self.semantic_field_required: query_args["semantic_text_field"] = self.semantic_text_field
            if self.method_type == 'hybrid_rerank':
                query_args["rerank_content_field"] = self.rerank_content_field
                query_args.setdefault("rank_window_size", 10);
                query_args.setdefault("min_score", 0.8)

            elasticsearch_results = self.query_function_ptr(**query_args)
        except Exception as e:
            raise ValueError(f"Error querying Elasticsearch with method {self.method_name}: {e}")

        prompt_for_model, raw_context_list = utils.create_openai_prompt(elasticsearch_results,
                                                                        self.primary_content_field_for_llm)
        generated_answer = utils.generate_openai_completion(
            openai_client=self.openai_client, model_deployment_name=self.openai_model_name,
            system_prompt=prompt_for_model, question=question_text
        )
        retrieved_context_display = "\n---\n".join(
            map(str, raw_context_list)) if raw_context_list else "No raw context extracted."
        return generated_answer, retrieved_context_display, elasticsearch_results, len(elasticsearch_results)

    def process_batch_question(self, question_data):
        question = question_data.get('question')
        if not question: return None
        elasticsearch_results = []
        try:
            query_args = {
                "es_client": self.es_client, "query": question, "index_name": self.index_name
            }
            if self.lexical_field_required: query_args["field_to_query"] = self.field_to_query
            if self.semantic_field_required: query_args["semantic_text_field"] = self.semantic_text_field
            if self.method_type == 'hybrid_rerank':
                query_args["rerank_content_field"] = self.rerank_content_field
                query_args.setdefault("rank_window_size", 10);
                query_args.setdefault("min_score", 0.8)

            elasticsearch_results = self.query_function_ptr(**query_args)
        except Exception as e:
            raise ValueError(f"Error querying ES for question '{question[:50]}...' with method {self.method_name}: {e}")

        prompt_for_model, retrieved_context_list = utils.create_openai_prompt(elasticsearch_results,
                                                                              self.primary_content_field_for_llm)
        generated_answer = utils.generate_openai_completion(
            openai_client=self.openai_client, model_deployment_name=self.openai_model_name,
            system_prompt=prompt_for_model, question=question
        )
        if generated_answer is None: return None
        processed_retrieved_context_str = ""
        if isinstance(prompt_for_model, str):
            context_marker = "Context:\n"
            if context_marker in prompt_for_model:
                processed_retrieved_context_str = prompt_for_model.split(context_marker, 1)[-1].split("\n<|user|>")[
                    0].strip()
        return {
            "question": question, "context": question_data.get('context'),
            "ref_answer": question_data.get('answer'),
            "retrieved_context_for_llm": [processed_retrieved_context_str],
            "raw_retrieved_contexts_from_es": retrieved_context_list,
            "generated_answer": generated_answer, "retrieval_method": self.method_name,
            "index_name_used": self.index_name,
            "lexical_field_queried": self.field_to_query,
            "semantic_text_field_queried": self.semantic_text_field,
            "primary_llm_content_field": self.primary_content_field_for_llm,
            "model_used": self.openai_model_name
        }


def setup_output_file_st(input_file_name, output_file_path_st, method_st, default_folder="rag_batch_outputs",
                         default_suffix="_answers.json"):
    if output_file_path_st:
        final_output_path = output_file_path_st
    else:
        base_name = os.path.splitext(os.path.basename(input_file_name))[0] if input_file_name else "batch_input"
        os.makedirs(default_folder, exist_ok=True)
        final_output_path = os.path.join(default_folder, f"{method_st.replace(' ', '_')}_{base_name}{default_suffix}")
    if os.path.exists(final_output_path):
        try:
            os.remove(final_output_path); st.info(
                f"Removed existing output for method '{method_st}': {final_output_path}")
        except OSError as e:
            st.warning(f"Could not remove {final_output_path}: {e}.")
    return final_output_path


@st.cache_data(ttl=300)
def get_cached_index_fields(_es_client, index_name, field_types):  # Changed es_client to _es_client
    """Cached function to get index fields. _es_client is not hashed."""
    if not _es_client or not index_name:  # Use _es_client internally
        return []
    mapping = utils.get_index_mapping(_es_client, index_name)  # Use _es_client internally
    if mapping and "properties" in mapping:
        return utils.extract_fields_from_mapping(mapping["properties"], field_types)
    return []


def show_rag_processor_tab(es_client, openai_client, azure_openai_model_name, default_index_name):
    st.header("🔎 RAG Question Processing")
    st.subheader("⚙️ RAG Processor Settings")

    index_name_input_rag = st.text_input(
        "Elasticsearch Index Name:",
        value=st.session_state.get("rag_proc_index_name", default_index_name),
        key="rag_index_input_tab",
        on_change=lambda: st.session_state.update(rag_proc_index_name=st.session_state.rag_index_input_tab,
                                                  rag_proc_fields_loaded=False)
    )
    st.session_state.rag_proc_index_name = index_name_input_rag

    text_keyword_field_options = []
    semantic_capable_field_options = []

    if 'rag_proc_fields_loaded' not in st.session_state:
        st.session_state.rag_proc_fields_loaded = False

    if es_client and index_name_input_rag and not st.session_state.rag_proc_fields_loaded:
        with st.spinner(f"Fetching fields for index '{index_name_input_rag}'..."):
            # Pass es_client to the cached function
            text_keyword_field_options = get_cached_index_fields(es_client, index_name_input_rag, ["text"])
            semantic_capable_field_options = get_cached_index_fields(es_client, index_name_input_rag,
                                                                     ["semantic_text"])

            st.session_state.rag_proc_text_keyword_fields_options = text_keyword_field_options
            st.session_state.rag_proc_semantic_capable_fields_options = semantic_capable_field_options
            st.session_state.rag_proc_fields_loaded = True

            st.session_state.rag_selected_lexical_field = None
            st.session_state.rag_selected_semantic_text_field = None
            st.session_state.rag_selected_primary_content_field = None
            st.session_state.rag_selected_rerank_content_field = None
            st.experimental_rerun()

    text_keyword_field_options = st.session_state.get('rag_proc_text_keyword_fields_options', [])
    semantic_capable_field_options = st.session_state.get('rag_proc_semantic_capable_fields_options', [])

    selected_lexical_field = st.selectbox(
        "Lexical Search Field:",
        options=[""] + text_keyword_field_options,
        index=0 if not st.session_state.get('rag_selected_lexical_field') else (
                    [""] + text_keyword_field_options).index(st.session_state.get('rag_selected_lexical_field')),
        key="rag_lexical_field_select",
        help="Select a 'text' or 'keyword' field for full-text search or the lexical part of hybrid queries."
    )
    st.session_state.rag_selected_lexical_field = selected_lexical_field if selected_lexical_field else None

    selected_semantic_text_field = st.selectbox(
        "Semantic Search Field:",
        options=[""] + semantic_capable_field_options,
        index=0 if not st.session_state.get('rag_selected_semantic_text_field') else (
                    [""] + semantic_capable_field_options).index(
            st.session_state.get('rag_selected_semantic_text_field')),
        key="rag_semantic_text_field_select",
        help="Select a 'dense_vector' or 'text' field (for models like ELSER) for semantic search or the semantic part of hybrid queries."
    )
    st.session_state.rag_selected_semantic_text_field = selected_semantic_text_field if selected_semantic_text_field else None

    primary_content_field_options = text_keyword_field_options
    selected_primary_content_field = st.selectbox(
        "Primary Content Field for LLM Prompt:",
        options=[""] + primary_content_field_options,
        index=0 if not st.session_state.get('rag_selected_primary_content_field') else (
                    [""] + primary_content_field_options).index(
            st.session_state.get('rag_selected_primary_content_field')),
        help="This field's content will be primarily used to construct the context for the LLM.",
        key="rag_primary_content_field_select"
    )
    st.session_state.rag_selected_primary_content_field = selected_primary_content_field if selected_primary_content_field else None

    selected_rerank_content_field = st.selectbox(
        "Content Field for Reranker Input (if using rerank method):",
        options=[""] + text_keyword_field_options,
        index=0 if not st.session_state.get('rag_selected_rerank_content_field') else (
                    [""] + text_keyword_field_options).index(st.session_state.get('rag_selected_rerank_content_field')),
        help="The content of this field will be sent to the reranker model.",
        key="rag_rerank_content_field_select"
    )
    st.session_state.rag_selected_rerank_content_field = selected_rerank_content_field if selected_rerank_content_field else None

    available_method_names = list(METHOD_INFO.keys())
    selected_method_names_rag = st.multiselect(
        "Select Retrieval Method(s):", options=available_method_names,
        default=[available_method_names[0]] if available_method_names else [],
        key="rag_methods_selector_tab"
    )

    st.divider();
    st.subheader("❓ Interactive Single Question")
    user_question = st.text_area("Type your question here:", height=100, key="single_question_input_tab")

    if st.button("🚀 Process Single Question", type="primary", use_container_width=True, key="process_single_q_tab"):
        valid_input = True
        if not user_question: st.warning("Please enter a question."); valid_input = False
        if not index_name_input_rag: st.warning("Please enter an Index Name."); valid_input = False
        if not selected_method_names_rag: st.warning("Please select retrieval method(s)."); valid_input = False
        if not selected_primary_content_field: st.warning(
            "Please select Primary Content Field for LLM."); valid_input = False

        for method_name in selected_method_names_rag:
            info = METHOD_INFO[method_name]
            if info.get('lexical_field_required') and not selected_lexical_field:
                st.warning(f"Method '{method_name}' requires a Lexical Search Field.");
                valid_input = False;
                break
            if info.get('semantic_field_required') and not selected_semantic_text_field:
                st.warning(f"Method '{method_name}' requires a Semantic Search Field.");
                valid_input = False;
                break
            if info['type'] == 'hybrid_rerank' and not selected_rerank_content_field:
                st.warning(f"Method '{method_name}' (rerank) requires Reranker Content Field.");
                valid_input = False;
                break

        if not openai_client or not es_client: st.error("Clients not initialized."); valid_input = False

        if valid_input:
            st.write(f"**Question:** {user_question}")
            for method_name_to_run in selected_method_names_rag:
                with st.expander(f"Results for Method: `{method_name_to_run}`", expanded=True):
                    try:
                        with st.spinner(f"Processing with method '{method_name_to_run}'..."):
                            processor = QuestionProcessor(
                                es_client, openai_client, azure_openai_model_name,
                                method_name_to_run, index_name_input_rag,
                                field_to_query=selected_lexical_field,
                                semantic_text_field=selected_semantic_text_field,
                                primary_content_field_for_llm=selected_primary_content_field,
                                rerank_content_field=selected_rerank_content_field
                            )
                            generated_answer, retrieved_context_str, es_results_json, num_es_results = processor.process_single_question(
                                user_question)
                        st.info(f"Retrieved **{num_es_results}** results from ES for method `{method_name_to_run}`.")
                        if es_results_json:
                            with st.popover("View Elasticsearch Results (JSON)"): st.json(es_results_json)
                        st.markdown("---")
                        if generated_answer:
                            st.subheader("✅ Generated Answer"); st.markdown(generated_answer)
                        else:
                            st.error("Could not generate an answer for this method.")
                        if retrieved_context_str is not None:
                            st.subheader("📚 Retrieved Context for LLM")
                            st.text_area(f"Context ({method_name_to_run}):", retrieved_context_str, height=150,
                                         disabled=False, key=f"s_ctx_{method_name_to_run}_{user_question[:10]}")
                    except ValueError as ve:
                        st.error(f"Error for method '{method_name_to_run}': {ve}")
                    except Exception as e:
                        st.error(f"Unexpected error with method '{method_name_to_run}': {e}"); st.code(
                            traceback.format_exc())
                st.markdown("---")

    st.divider();
    st.subheader("📄 Batch Processing from File")
    uploaded_file_batch = st.file_uploader("Upload JSON questions file", type=['json'], key="batch_file_uploader_tab")

    if st.button("📂 Process Batch File with Selected Method(s)", use_container_width=True,
                 key="process_batch_file_tab"):
        valid_input = True
        if uploaded_file_batch is None: st.warning("Please upload a JSON file."); valid_input = False
        if not index_name_input_rag: st.warning("Please enter an Index Name."); valid_input = False
        if not selected_method_names_rag: st.warning("Please select retrieval method(s)."); valid_input = False
        if not selected_primary_content_field: st.warning(
            "Please select Primary Content Field for LLM."); valid_input = False
        for method_name in selected_method_names_rag:
            info = METHOD_INFO[method_name]
            if info.get('lexical_field_required') and not selected_lexical_field:
                st.warning(f"Method '{method_name}' requires a Lexical Search Field.");
                valid_input = False;
                break
            if info.get('semantic_field_required') and not selected_semantic_text_field:
                st.warning(f"Method '{method_name}' requires a Semantic Search Field.");
                valid_input = False;
                break
            if info['type'] == 'hybrid_rerank' and not selected_rerank_content_field:
                st.warning(f"Method '{method_name}' (rerank) requires Reranker Content Field.");
                valid_input = False;
                break
        if not openai_client or not es_client: st.error("Clients not initialized."); valid_input = False

        if valid_input:
            try:
                uploaded_file_batch.seek(0);
                questions_data = json.load(uploaded_file_batch)
                if not isinstance(questions_data, list) or not questions_data:
                    st.error("Uploaded file must be a non-empty list of questions.");
                    return
            except Exception as e:
                st.error(f"Error reading/parsing file: {e}"); return

            input_filename_for_default = uploaded_file_batch.name;
            generated_files_info = []
            for method_name_to_run in selected_method_names_rag:
                st.markdown(f"--- \n#### Processing with Method: `{method_name_to_run}`")
                actual_output_file = setup_output_file_st(input_filename_for_default, None, method_name_to_run,
                                                          default_folder="rag_batch_outputs")
                st.info(
                    f"Processing {len(questions_data)}q. Results for '{method_name_to_run}' -> '{actual_output_file}'")
                try:
                    processor = QuestionProcessor(
                        es_client, openai_client, azure_openai_model_name,
                        method_name_to_run, index_name_input_rag,
                        field_to_query=selected_lexical_field,
                        semantic_text_field=selected_semantic_text_field,
                        primary_content_field_for_llm=selected_primary_content_field,
                        rerank_content_field=selected_rerank_content_field
                    )
                except ValueError as ve:
                    st.error(f"Config error for '{method_name_to_run}': {ve}. Skipping."); continue

                results_summary_for_method = [];
                all_results_for_this_method_file = []
                progress_bar = st.progress(0, text=f"Progress for '{method_name_to_run}'")
                status_text = st.empty()
                for i, question_data_item in enumerate(questions_data):
                    status_text.text(f"'{method_name_to_run}': Question {i + 1}/{len(questions_data)}...")
                    try:
                        result = processor.process_batch_question(question_data_item)
                        if result:
                            all_results_for_this_method_file.append(result)
                            results_summary_for_method.append({"question": result["question"],
                                                               "answer_preview": result.get("generated_answer", "N/A")[
                                                                                 :100] + "..."})
                    except ValueError as ve:
                        st.warning(f"Skipping Q{i + 1} for '{method_name_to_run}': {ve}")
                    except Exception as e:
                        st.error(f"Error on Q{i + 1} for '{method_name_to_run}': {e}")
                    progress_bar.progress((i + 1) / len(questions_data))

                if all_results_for_this_method_file:
                    with open(actual_output_file, 'w') as f_out:
                        json.dump(all_results_for_this_method_file, f_out, indent=4)
                    status_text.success(f"Batch for '{method_name_to_run}' complete! Saved to '{actual_output_file}'.")
                    generated_files_info.append({"method": method_name_to_run, "path": actual_output_file,
                                                 "summary": results_summary_for_method})
                else:
                    status_text.warning(f"No results for method '{method_name_to_run}'.")

            st.markdown("--- \n### Batch Processing Summary & Downloads")
            if generated_files_info:
                for file_info in generated_files_info:
                    st.subheader(f"Results for Method: `{file_info.get('method', 'Unknown')}`")
                    summary_data = file_info.get('summary')
                    if isinstance(summary_data, list) and summary_data:
                        try:
                            st.dataframe(pd.DataFrame(summary_data).head())
                        except Exception as e:
                            st.error(f"Error displaying summary for '{file_info.get('method')}': {e}")
                    else:
                        st.caption(f"No summary for '{file_info.get('method')}'.")
                    file_path = file_info.get('path')
                    if file_path and os.path.exists(file_path):
                        try:
                            with open(file_path, "rb") as fp:
                                st.download_button(
                                    label=f"Download for '{file_info.get('method')}' ({os.path.basename(file_path)})",
                                    data=fp, file_name=os.path.basename(file_path), mime="application/json",
                                    key=f"dl_batch_{file_info.get('method', 'na').replace(' ', '_')}_{os.path.basename(file_path)}")
                        except Exception as e:
                            st.error(f"Download error for {file_path}: {e}")
                    elif file_path:
                        st.warning(f"Output file not found: {file_path}")
            else:
                st.warning("No output files generated.")
