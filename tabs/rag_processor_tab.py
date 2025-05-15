# tabs/rag_processor_tab.py
import streamlit as st
import os
import json
import pandas as pd
import retriever_utils as utils
import traceback

# --- Configuration for retrieval methods ---
METHOD_INFO = {
    'Full-Text Search': {'func': utils.query_fulltext, 'type': 'fulltext', 'lexical_field_required': True,
                         'semantic_field_required': False},
    'Semantic Search': {'func': utils.query_semantic, 'type': 'semantic', 'lexical_field_required': False,
                        'semantic_field_required': True},
    'Hybrid Search (RRF)': {'func': utils.query_hybrid, 'type': 'hybrid', 'lexical_field_required': True,
                            'semantic_field_required': True, 'uses_rrf': True},
    'Hybrid Search (Linear Combination)': {'func': utils.query_hybrid_linear, 'type': 'hybrid',
                                           'lexical_field_required': True, 'semantic_field_required': True,
                                           'uses_rrf': False},
    'BGE Semantic Search': {'func': utils.query_semantic_bge, 'type': 'semantic', 'lexical_field_required': False,
                            'semantic_field_required': True, 'vector_field_prefix': 'content_semantic_bge-m3'},
    'BGE Hybrid Search (RRF)': {'func': utils.query_hybrid_bge, 'type': 'hybrid', 'lexical_field_required': True,
                                'semantic_field_required': True, 'vector_field_prefix': 'content_semantic_bge-m3',
                                'uses_rrf': True},
    'BGE Hybrid Search (Linear)': {'func': utils.query_hybrid_linear_bge, 'type': 'hybrid',
                                   'lexical_field_required': True, 'semantic_field_required': True,
                                   'vector_field_prefix': 'content_semantic_bge-m3', 'uses_rrf': False},
    'Hybrid Rerank (Cohere)': {'func': utils.query_hybrid_rerank, 'type': 'hybrid_rerank',
                               'lexical_field_required': True, 'semantic_field_required': True, 'uses_rrf_base': True,
                               'requires_reranker_id': True},
    'BGE Hybrid Rerank (Cohere)': {'func': utils.query_hybrid_bge_rerank, 'type': 'hybrid_rerank',
                                   'lexical_field_required': True, 'semantic_field_required': True,
                                   'vector_field_prefix': 'content_semantic_bge-m3', 'uses_rrf_base': True,
                                   'requires_reranker_id': True}
}


class QuestionProcessor:
    def __init__(self, es_client, openai_client, openai_model_name, method_name, index_name,
                 lexical_field,
                 semantic_text_field,
                 primary_content_field_for_llm,
                 rrf_window_size, rrf_rank_constant,
                 rerank_content_field=None,
                 reranker_inference_id=None,
                 base_rrf_window_size_for_rerank=50,
                 base_rrf_rank_constant_for_rerank=60
                 ):
        self.es_client = es_client
        self.openai_client = openai_client
        self.openai_model_name = openai_model_name
        self.method_name = method_name
        self.index_name = index_name
        self.lexical_field = lexical_field
        self.semantic_text_field = semantic_text_field
        self.primary_content_field_for_llm = primary_content_field_for_llm
        self.rerank_content_field = rerank_content_field if rerank_content_field else primary_content_field_for_llm

        self.rrf_window_size = rrf_window_size
        self.rrf_rank_constant = rrf_rank_constant
        self.reranker_inference_id = reranker_inference_id
        self.base_rrf_window_size_for_rerank = base_rrf_window_size_for_rerank
        self.base_rrf_rank_constant_for_rerank = base_rrf_rank_constant_for_rerank

        if self.method_name not in METHOD_INFO:
            raise ValueError(f"Unknown method '{self.method_name}' selected.")
        self.query_function_ptr = METHOD_INFO[self.method_name]['func']
        self.method_type = METHOD_INFO[self.method_name]['type']
        self.lexical_field_required = METHOD_INFO[self.method_name]['lexical_field_required']
        self.semantic_field_required = METHOD_INFO[self.method_name]['semantic_field_required']
        self.uses_rrf = METHOD_INFO[self.method_name].get('uses_rrf', False)
        self.uses_rrf_base_for_rerank = METHOD_INFO[self.method_name].get('uses_rrf_base', False)
        self.requires_reranker_id = METHOD_INFO[self.method_name].get('requires_reranker_id', False)

    def process_single_question(self, question_text):
        elasticsearch_results = []
        try:
            query_args = {
                "es_client": self.es_client, "query": question_text, "index_name": self.index_name
            }
            if self.lexical_field_required: query_args["field_to_query"] = self.lexical_field
            if self.semantic_field_required: query_args["semantic_text_field"] = self.semantic_text_field

            if self.uses_rrf:
                query_args["rank_window_size"] = self.rrf_window_size
                query_args["rank_constant"] = self.rrf_rank_constant

            if self.method_type == 'hybrid_rerank':
                query_args["rerank_content_field"] = self.rerank_content_field
                query_args["inference_id"] = self.reranker_inference_id
                if self.uses_rrf_base_for_rerank:
                    query_args["base_rrf_window_size"] = self.base_rrf_window_size_for_rerank
                    query_args["base_rrf_rank_constant"] = self.base_rrf_rank_constant_for_rerank
                query_args.setdefault("rank_window_size", 10)
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
            if self.lexical_field_required: query_args["field_to_query"] = self.lexical_field
            if self.semantic_field_required: query_args["semantic_text_field"] = self.semantic_text_field

            if self.uses_rrf:
                query_args["rank_window_size"] = self.rrf_window_size
                query_args["rank_constant"] = self.rrf_rank_constant

            if self.method_type == 'hybrid_rerank':
                query_args["rerank_content_field"] = self.rerank_content_field
                query_args["inference_id"] = self.reranker_inference_id
                if self.uses_rrf_base_for_rerank:
                    query_args["base_rrf_window_size"] = self.base_rrf_window_size_for_rerank
                    query_args["base_rrf_rank_constant"] = self.base_rrf_rank_constant_for_rerank
                query_args.setdefault("rank_window_size", 10)
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

        output_dict = {
            "question": question, "context": question_data.get('context'),
            "ref_answer": question_data.get('answer'),
            "retrieved_context_for_llm": [processed_retrieved_context_str],
            "raw_retrieved_contexts_from_es": retrieved_context_list,
            "generated_answer": generated_answer, "retrieval_method": self.method_name,
            "index_name_used": self.index_name,
            "lexical_field_queried": self.lexical_field,
            "semantic_text_field_queried": self.semantic_text_field,
            "primary_llm_content_field": self.primary_content_field_for_llm,
            "model_used": self.openai_model_name
        }
        if self.uses_rrf or self.uses_rrf_base_for_rerank:
            output_dict[
                "rrf_rank_window_size"] = self.rrf_window_size if self.uses_rrf else self.base_rrf_window_size_for_rerank
            output_dict[
                "rrf_rank_constant"] = self.rrf_rank_constant if self.uses_rrf else self.base_rrf_rank_constant_for_rerank
        if self.requires_reranker_id:
            output_dict["reranker_inference_id_used"] = self.reranker_inference_id
        return output_dict


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
def get_cached_index_fields(_es_client, index_name, field_types):
    if not _es_client or not index_name: return []
    mapping = utils.get_index_mapping(_es_client, index_name)
    if mapping and "properties" in mapping:
        return utils.extract_fields_from_mapping(mapping["properties"], field_types)
    return []


@st.cache_data(ttl=300)
def get_cached_reranker_ids(_es_client, app_config_for_http: dict):
    if not _es_client and not app_config_for_http.get("ES_URL"):
        return []
    return utils.get_rerank_inference_ids(_es_client, app_config_for_http)


def show_rag_processor_tab(es_client, openai_client, azure_openai_model_name, default_index_name):
    st.header("🔎 RAG Question Processing")

    with st.container(border=True):
        st.subheader("⚙️ RAG Processor Settings")

        index_name_input_rag = st.text_input(
            "Elasticsearch Index Name:",
            value=st.session_state.get("rag_proc_index_name", default_index_name),
            key="rag_index_input_tab",
            on_change=lambda: st.session_state.update(
                rag_proc_index_name=st.session_state.rag_index_input_tab,
                rag_proc_fields_loaded=False,
                rag_proc_rerankers_loaded=False
            )
        )
        st.session_state.rag_proc_index_name = index_name_input_rag

        if 'rag_proc_fields_loaded' not in st.session_state:
            st.session_state.rag_proc_fields_loaded = False
        if 'rag_proc_rerankers_loaded' not in st.session_state:
            st.session_state.rag_proc_rerankers_loaded = False

        app_config = st.session_state.get("app_config", {})

        needs_rerun_after_loading = False
        if es_client and index_name_input_rag:
            if not st.session_state.rag_proc_fields_loaded:
                with st.spinner(f"Fetching fields for index '{index_name_input_rag}'..."):
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
                    needs_rerun_after_loading = True

        if not st.session_state.rag_proc_rerankers_loaded:
            if es_client or app_config.get("ES_URL"):
                with st.spinner("Fetching reranker models..."):
                    reranker_id_options = get_cached_reranker_ids(es_client, app_config)
                    st.session_state.rag_proc_reranker_id_options = reranker_id_options
                    st.session_state.rag_proc_rerankers_loaded = True
                    # Set default for selectbox, but override will take precedence if user types
                    st.session_state.rag_selectbox_reranker_id = reranker_id_options[0] if reranker_id_options else None
                    st.session_state.rag_text_input_reranker_id = ""  # Initialize override field
                    needs_rerun_after_loading = True
            elif index_name_input_rag:
                st.caption("ES client not available and ES_URL not in config; cannot fetch reranker models.")

        if needs_rerun_after_loading:
            st.rerun()

        text_keyword_field_options = st.session_state.get('rag_proc_text_keyword_fields_options', [])
        semantic_capable_field_options = st.session_state.get('rag_proc_semantic_capable_fields_options', [])
        reranker_id_options = st.session_state.get('rag_proc_reranker_id_options', [])

        col_lex, col_sem = st.columns(2)
        with col_lex:
            selected_lexical_field = st.selectbox(
                "Lexical Search Field:", options=[""] + text_keyword_field_options,
                index=0 if not st.session_state.get('rag_selected_lexical_field') else (
                            [""] + text_keyword_field_options).index(
                    st.session_state.get('rag_selected_lexical_field')),
                key="rag_lexical_field_select", help="Field for full-text/lexical part of queries."
            )
            st.session_state.rag_selected_lexical_field = selected_lexical_field if selected_lexical_field else None
        with col_sem:
            selected_semantic_text_field = st.selectbox(
                "Semantic Search Field:", options=[""] + semantic_capable_field_options,
                index=0 if not st.session_state.get('rag_selected_semantic_text_field') else (
                            [""] + semantic_capable_field_options).index(
                    st.session_state.get('rag_selected_semantic_text_field')),
                key="rag_semantic_text_field_select", help="Field for semantic/vector part of queries."
            )
            st.session_state.rag_selected_semantic_text_field = selected_semantic_text_field if selected_semantic_text_field else None

        selected_primary_content_field = st.selectbox(
            "Primary Content Field for LLM Prompt:", options=[""] + text_keyword_field_options,
            index=0 if not st.session_state.get('rag_selected_primary_content_field') else (
                        [""] + text_keyword_field_options).index(
                st.session_state.get('rag_selected_primary_content_field')),
            key="rag_primary_content_field_select", help="Field content used to build LLM context."
        )
        st.session_state.rag_selected_primary_content_field = selected_primary_content_field if selected_primary_content_field else None

        with st.expander("Reranker Settings (if using rerank methods)"):
            selected_rerank_content_field = st.selectbox(
                "Content Field for Reranker Input:", options=[""] + text_keyword_field_options,
                index=0 if not st.session_state.get('rag_selected_rerank_content_field') else (
                            [""] + text_keyword_field_options).index(
                    st.session_state.get('rag_selected_rerank_content_field')),
                key="rag_rerank_content_field_select", help="Field content sent to reranker model."
            )
            st.session_state.rag_selected_rerank_content_field = selected_rerank_content_field if selected_rerank_content_field else None

            if not reranker_id_options and (
                    es_client or app_config.get("ES_URL")) and index_name_input_rag and st.session_state.get(
                    'rag_proc_rerankers_loaded', False):
                st.caption(
                    "No 'rerank' type inference models found in Elasticsearch or an error occurred fetching them. You can manually enter an ID below.")

            # Dropdown for discovered rerankers
            selectbox_reranker_id = st.selectbox(
                "Select Reranker Model (Inference ID):", options=[""] + reranker_id_options,
                index=0 if not st.session_state.get('rag_selectbox_reranker_id') else (
                            [""] + reranker_id_options).index(st.session_state.get('rag_selectbox_reranker_id')),
                key="rag_selectbox_reranker_id_select",  # New key for selectbox
                help="Select a discovered reranker or use the override field below."
            )
            st.session_state.rag_selectbox_reranker_id = selectbox_reranker_id if selectbox_reranker_id else None

            # Text input for override
            text_input_reranker_id = st.text_input(
                "Override Reranker Inference ID (optional):",
                value=st.session_state.get('rag_text_input_reranker_id', ""),
                key="rag_text_input_reranker_id_override",
                help="Manually enter an Inference ID. This will override the dropdown selection."
            )
            st.session_state.rag_text_input_reranker_id = text_input_reranker_id.strip()

            # Determine the final reranker ID to use
            final_selected_reranker_id = st.session_state.rag_text_input_reranker_id if st.session_state.rag_text_input_reranker_id else st.session_state.rag_selectbox_reranker_id
            st.session_state.rag_selected_reranker_inference_id = final_selected_reranker_id  # This is what gets passed to QuestionProcessor

        with st.expander("RRF Parameters (Advanced)"):
            st.markdown("##### For RRF-based Hybrid Methods")
            col_rrf1, col_rrf2 = st.columns(2)
            with col_rrf1:
                rrf_window_size_ui = st.number_input("RRF: Rank Window Size", min_value=1,
                                                     value=st.session_state.get('rrf_window_size', 50), step=1,
                                                     key="rrf_window_input")
                st.session_state.rrf_window_size = rrf_window_size_ui
            with col_rrf2:
                rrf_rank_constant_ui = st.number_input("RRF: Rank Constant", min_value=1,
                                                       value=st.session_state.get('rrf_rank_constant', 60), step=1,
                                                       key="rrf_constant_input")
                st.session_state.rrf_rank_constant = rrf_rank_constant_ui
            st.markdown("##### For Reranker's Base Retriever (if it uses RRF)")
            col_rrf_base1, col_rrf_base2 = st.columns(2)
            with col_rrf_base1:
                base_rrf_window_ui = st.number_input("Reranker Base RRF: Window Size", min_value=1,
                                                     value=st.session_state.get('base_rrf_window_size_for_rerank', 50),
                                                     step=1, key="base_rrf_window_rerank_input")
                st.session_state.base_rrf_window_size_for_rerank = base_rrf_window_ui
            with col_rrf_base2:
                base_rrf_constant_ui = st.number_input("Reranker Base RRF: Rank Constant", min_value=1,
                                                       value=st.session_state.get('base_rrf_rank_constant_for_rerank',
                                                                                  60), step=1,
                                                       key="base_rrf_constant_rerank_input")
                st.session_state.base_rrf_rank_constant_for_rerank = base_rrf_constant_ui

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
            if info.get('lexical_field_required') and not st.session_state.rag_selected_lexical_field:
                st.warning(f"Method '{method_name}' requires a Lexical Search Field.");
                valid_input = False;
                break
            if info.get('semantic_field_required') and not st.session_state.rag_selected_semantic_text_field:
                st.warning(f"Method '{method_name}' requires a Semantic Search Field.");
                valid_input = False;
                break
            if info.get('requires_reranker_id',
                        False) and not st.session_state.rag_selected_reranker_inference_id:  # Use the final determined ID
                st.warning(f"Method '{method_name}' requires a Reranker Model (Inference ID).");
                valid_input = False;
                break
            if info.get('requires_reranker_id',
                        False) and st.session_state.rag_selected_reranker_inference_id and not st.session_state.rag_selected_rerank_content_field:
                st.warning(
                    f"Method '{method_name}' (rerank) requires Reranker Content Field when a Reranker Model is selected.");
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
                                lexical_field=st.session_state.rag_selected_lexical_field,
                                semantic_text_field=st.session_state.rag_selected_semantic_text_field,
                                primary_content_field_for_llm=st.session_state.rag_selected_primary_content_field,
                                rrf_window_size=st.session_state.rrf_window_size,
                                rrf_rank_constant=st.session_state.rrf_rank_constant,
                                rerank_content_field=st.session_state.rag_selected_rerank_content_field,
                                reranker_inference_id=st.session_state.rag_selected_reranker_inference_id,
                                # Pass the final ID
                                base_rrf_window_size_for_rerank=st.session_state.base_rrf_window_size_for_rerank,
                                base_rrf_rank_constant_for_rerank=st.session_state.base_rrf_rank_constant_for_rerank
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
            if info.get('lexical_field_required') and not st.session_state.rag_selected_lexical_field:
                st.warning(f"Method '{method_name}' requires a Lexical Search Field.");
                valid_input = False;
                break
            if info.get('semantic_field_required') and not st.session_state.rag_selected_semantic_text_field:
                st.warning(f"Method '{method_name}' requires a Semantic Search Field.");
                valid_input = False;
                break
            if info.get('requires_reranker_id',
                        False) and not st.session_state.rag_selected_reranker_inference_id:  # Use the final determined ID
                st.warning(f"Method '{method_name}' requires a Reranker Model (Inference ID).");
                valid_input = False;
                break
            if info.get('requires_reranker_id',
                        False) and st.session_state.rag_selected_reranker_inference_id and not st.session_state.rag_selected_rerank_content_field:
                st.warning(
                    f"Method '{method_name}' (rerank) requires Reranker Content Field when a Reranker Model is selected.");
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
                        lexical_field=st.session_state.rag_selected_lexical_field,
                        semantic_text_field=st.session_state.rag_selected_semantic_text_field,
                        primary_content_field_for_llm=st.session_state.rag_selected_primary_content_field,
                        rrf_window_size=st.session_state.rrf_window_size,
                        rrf_rank_constant=st.session_state.rrf_rank_constant,
                        rerank_content_field=st.session_state.rag_selected_rerank_content_field,
                        reranker_inference_id=st.session_state.rag_selected_reranker_inference_id,  # Pass the final ID
                        base_rrf_window_size_for_rerank=st.session_state.base_rrf_window_size_for_rerank,
                        base_rrf_rank_constant_for_rerank=st.session_state.base_rrf_rank_constant_for_rerank
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
