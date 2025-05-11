# tabs/ground_truth_tab.py
import streamlit as st
import pandas as pd
import json
import random
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import prompts # For default prompts fallback
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0

# Define keys for session state prompts (must match those in prompt_settings_tab.py and app.py)
PROMPT_SESSION_KEYS = {
    "QA Generation": "session_qa_generation_prompt",
    "Question Groundedness Critique": "session_q_groundedness_prompt",
    "Question Relevance Critique": "session_q_relevance_prompt",
    "Question Standalone Critique": "session_q_standalone_prompt",
    "Translation Task": "session_translate_prompt"
}

# Helper function to get current prompt from session state or default
def get_current_prompt(session_key, default_template):
    return st.session_state.get(session_key, default_template)

def call_llm_for_translation(client, model_deployment_name, text_to_translate, target_language):
    translate_prompt_template = get_current_prompt(PROMPT_SESSION_KEYS["Translation Task"], prompts.TRANSLATE_PROMPT_TEMPLATE)
    translation_prompt_text = translate_prompt_template.format(
        target_language=target_language,
        text_to_translate=text_to_translate
    )
    try:
        response = client.chat.completions.create(
            model=model_deployment_name,
            messages=[{"role": "system", "content": "You are a helpful translation assistant."}, {"role": "user", "content": translation_prompt_text}],
            max_tokens=len(text_to_translate) * 2 + 150 # Adjusted max_tokens slightly
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error translating text to {target_language}: {e}. Falling back to English.")
        return text_to_translate

def call_llm_for_gt(client, model_deployment_name, prompt_content):
    try:
        response = client.chat.completions.create(
            model=model_deployment_name,
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt_content}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM for GT: {e}"); return None

def run_ground_truth_generation(
        uploaded_file, num_generations_to_create, azure_llm_client,
        azure_model_name, progress_bar_placeholder, status_text_placeholder
):
    if uploaded_file is None: st.error("Please upload a raw data file."); return None, []
    try:
        uploaded_file.seek(0); file_content_bytes = uploaded_file.read()
        try: file_content_str = file_content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            st.warning("UTF-8 decoding failed, trying 'latin-1'"); file_content_str = file_content_bytes.decode("latin-1", errors='replace')
        raw_data = json.loads(file_content_str)
    except Exception as e: st.error(f"Error reading or parsing JSON file: {e}"); return None, []

    detected_lang = "en"; sample_text_for_detection = ""
    if raw_data and isinstance(raw_data, list) and raw_data[0].get('content'):
        sample_text_for_detection = raw_data[0]['content'][:1000]
    if sample_text_for_detection:
        try: detected_lang = detect(sample_text_for_detection); st.info(f"Detected document language: **{detected_lang.upper()}**")
        except LangDetectException: st.warning("Could not reliably detect document language. Defaulting to English."); detected_lang = "en"
    else: st.warning("No content found for language detection. Defaulting to English.")

    # Get current prompts from session state or defaults
    qa_gen_prompt_template = get_current_prompt(PROMPT_SESSION_KEYS["QA Generation"], prompts.QA_GENERATION_PROMPT_TEMPLATE)
    q_groundedness_prompt_template = get_current_prompt(PROMPT_SESSION_KEYS["Question Groundedness Critique"], prompts.QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT_TEMPLATE)
    q_relevance_prompt_template = get_current_prompt(PROMPT_SESSION_KEYS["Question Relevance Critique"], prompts.QUESTION_RELEVANCE_CRITIQUE_PROMPT_TEMPLATE)
    q_standalone_prompt_template = get_current_prompt(PROMPT_SESSION_KEYS["Question Standalone Critique"], prompts.QUESTION_STANDALONE_CRITIQUE_PROMPT_TEMPLATE)

    if detected_lang != "en":
        with st.spinner(f"Translating prompts to {detected_lang.upper()}..."):
            qa_gen_prompt_template = call_llm_for_translation(azure_llm_client, azure_model_name, qa_gen_prompt_template, detected_lang)
            q_groundedness_prompt_template = call_llm_for_translation(azure_llm_client, azure_model_name, q_groundedness_prompt_template, detected_lang)
            q_relevance_prompt_template = call_llm_for_translation(azure_llm_client, azure_model_name, q_relevance_prompt_template, detected_lang)
            q_standalone_prompt_template = call_llm_for_translation(azure_llm_client, azure_model_name, q_standalone_prompt_template, detected_lang)
        st.success(f"Prompts translated to {detected_lang.upper()}.")

    langchain_documents = []
    for i, item in enumerate(raw_data):
        if 'content' in item:
            metadata = item.get('metadata', {}); metadata['source'] = uploaded_file.name; metadata['original_index'] = i
            langchain_documents.append(LangchainDocument(page_content=item['content'], metadata=metadata))
        else: st.warning(f"Item at index {i} in the JSON file is missing a 'content' field and will be skipped.")
    if not langchain_documents: st.error("No documents with 'content' field found."); return None, []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, add_start_index=True, separators=["\n\n", "\n", ".", " ", ""],)
    docs_processed = [];
    for doc in langchain_documents: docs_processed.extend(text_splitter.split_documents([doc]))
    st.info(f"Loaded and processed {len(docs_processed)} document chunks.")
    if not docs_processed: st.error("No document chunks could be processed."); return None, []

    actual_n_generations = min(num_generations_to_create, len(docs_processed))
    if num_generations_to_create > len(docs_processed) and len(docs_processed) > 0 :
        st.warning(f"Requested {num_generations_to_create} Q&A pairs, but only {len(docs_processed)} available. Will generate {len(docs_processed)}.")
    if actual_n_generations == 0 and len(docs_processed) > 0: actual_n_generations = len(docs_processed)
    elif actual_n_generations == 0: st.warning("Number of generations is zero or no documents to process."); return pd.DataFrame(), []

    st.info(f"Attempting to generate {actual_n_generations} Q&A couples...")
    outputs_qa = []; progress_bar_placeholder.progress(0)
    sampled_contexts = random.sample(docs_processed, actual_n_generations) if actual_n_generations > 0 and actual_n_generations <= len(docs_processed) else docs_processed

    for i, sampled_context in enumerate(sampled_contexts):
        status_text_placeholder.text(f"Generating Q&A pair {i + 1}/{actual_n_generations}...")
        try:
            qa_prompt_formatted = qa_gen_prompt_template.format(context=sampled_context.page_content)
            output_qa_couple_str = call_llm_for_gt(azure_llm_client, azure_model_name, qa_prompt_formatted)
            if output_qa_couple_str and "Factoid question:" in output_qa_couple_str and "Answer:" in output_qa_couple_str:
                parts = output_qa_couple_str.split("Factoid question:")[-1].split("Answer:")
                question = parts[0].strip(); answer = parts[1].strip()
                if len(answer) >= 300: st.warning(f"Skipping Q&A pair {i + 1} due to long answer."); continue
                outputs_qa.append({"context": sampled_context.page_content, "question": question, "answer": answer, "source_doc": sampled_context.metadata.get("source", "unknown_source")})
            else: st.warning(f"Could not parse Q&A from LLM response for context {i + 1}. Response: {output_qa_couple_str[:200]}...")
        except Exception as e: st.error(f"Error generating Q&A for document chunk {i + 1}: {e}")
        if actual_n_generations > 0: progress_bar_placeholder.progress((i + 1) / actual_n_generations)

    status_text_placeholder.text(f"Generated {len(outputs_qa)} Q&A pairs.")
    if not outputs_qa: st.error("No Q&A pairs were successfully generated."); progress_bar_placeholder.progress(1.0); return pd.DataFrame(), []

    st.info(f"Generating critiques for {len(outputs_qa)} QA couples...")
    critique_progress_count = 0; critique_progress_bar = st.progress(0, text="Critique Progress")
    for i, output_item in enumerate(outputs_qa):
        status_text_placeholder.text(f"Critiquing Q&A pair {i + 1}/{len(outputs_qa)}...")
        try: # Groundedness
            groundedness_prompt_formatted = q_groundedness_prompt_template.format(context=output_item["context"], question=output_item["question"])
            eval_str = call_llm_for_gt(azure_llm_client, azure_model_name, groundedness_prompt_formatted)
            if eval_str and "Total rating:" in eval_str and "Evaluation:" in eval_str:
                output_item["groundedness_score"] = int(eval_str.split("Total rating:")[-1].strip())
                output_item["groundedness_eval"] = eval_str.split("Total rating:")[-2].split("Evaluation:")[-1].strip()
            else: output_item["groundedness_score"] = 0
        except Exception as e: st.warning(f"Groundedness critique error: {e}"); output_item["groundedness_score"] = 0
        critique_progress_count += 1
        try: # Relevance
            relevance_prompt_formatted = q_relevance_prompt_template.format(question=output_item["question"])
            eval_str = call_llm_for_gt(azure_llm_client, azure_model_name, relevance_prompt_formatted)
            if eval_str and "Total rating:" in eval_str and "Evaluation:" in eval_str:
                output_item["relevance_score"] = int(eval_str.split("Total rating:")[-1].strip())
                output_item["relevance_eval"] = eval_str.split("Total rating:")[-2].split("Evaluation:")[-1].strip()
            else: output_item["relevance_score"] = 0
        except Exception as e: st.warning(f"Relevance critique error: {e}"); output_item["relevance_score"] = 0
        critique_progress_count += 1
        try: # Standalone
            standalone_prompt_formatted = q_standalone_prompt_template.format(question=output_item["question"])
            eval_str = call_llm_for_gt(azure_llm_client, azure_model_name, standalone_prompt_formatted)
            if eval_str and "Total rating:" in eval_str and "Evaluation:" in eval_str:
                output_item["standalone_score"] = int(eval_str.split("Total rating:")[-1].strip())
                output_item["standalone_eval"] = eval_str.split("Total rating:")[-2].split("Evaluation:")[-1].strip()
            else: output_item["standalone_score"] = 0
        except Exception as e: st.warning(f"Standalone critique error: {e}"); output_item["standalone_score"] = 0
        critique_progress_count += 1
        if len(outputs_qa) > 0: critique_progress_bar.progress(critique_progress_count / (len(outputs_qa) * 3))

    status_text_placeholder.text("Critiques generated. Filtering results...")
    generated_questions_df = pd.DataFrame(outputs_qa)
    for col in ["groundedness_score", "relevance_score", "standalone_score"]:
        if col not in generated_questions_df.columns: generated_questions_df[col] = 0
    st.subheader("Generated Q&A (Before Filtering)")
    st.dataframe(generated_questions_df[["question", "answer", "groundedness_score", "relevance_score", "standalone_score"]].head())
    filtered_df = generated_questions_df.loc[(generated_questions_df["groundedness_score"] >= 4) & (generated_questions_df["relevance_score"] >= 3) & (generated_questions_df["standalone_score"] >= 2)]
    status_text_placeholder.text("Ground truth generation complete."); progress_bar_placeholder.progress(1.0); critique_progress_bar.progress(1.0)
    return filtered_df, outputs_qa

def show_ground_truth_tab(openai_client, azure_openai_model_name):
    st.header("📝 Ground Truth Generation for RAG Evaluation")
    if not openai_client or not azure_openai_model_name:
        st.error("OpenAI client or model name not configured."); return
    uploaded_raw_data_gt = st.file_uploader("Upload Raw Data JSON File", type="json", key="gt_raw_data_uploader")
    output_gt_filename_suggestion = "generated_ground_truth.json"
    if uploaded_raw_data_gt is not None:
        base, ext = os.path.splitext(uploaded_raw_data_gt.name)
        output_gt_filename_suggestion = f"{base}_ground_truth.json"
    output_gt_filename = st.text_input("Output Filename", value=output_gt_filename_suggestion, key="gt_output_filename")
    num_generations_gt = st.number_input("Number of Q&A couples", min_value=0, value=10, step=1, key="gt_num_generations")
    if num_generations_gt == 0: st.caption("0 will process all chunks.")
    gt_progress_bar_overall = st.progress(0, text="Overall Q&A Generation Progress")
    gt_status_text = st.empty()
    if st.button("Generate Ground Truth Data", key="gt_generate_button", use_container_width=True):
        if uploaded_raw_data_gt:
            gt_status_text.text("Starting ground truth generation process...")
            filtered_ground_truth_df, _ = run_ground_truth_generation(
                uploaded_file=uploaded_raw_data_gt, num_generations_to_create=num_generations_gt,
                azure_llm_client=openai_client, azure_model_name=azure_openai_model_name,
                progress_bar_placeholder=gt_progress_bar_overall, status_text_placeholder=gt_status_text
            )
            if filtered_ground_truth_df is not None and not filtered_ground_truth_df.empty:
                st.subheader("Final Filtered Evaluation Dataset")
                st.dataframe(filtered_ground_truth_df[["question", "answer", "context", "groundedness_score", "relevance_score", "standalone_score"]])
                gt_json_data = filtered_ground_truth_df.to_json(orient='records', indent=4)
                st.download_button("Download Filtered Ground Truth (JSON)", gt_json_data, output_gt_filename, "application/json", key="download_gt_filtered")
                st.success(f"Filtered data ready for download as {output_gt_filename}")
            elif filtered_ground_truth_df is not None: st.warning("No Q&A pairs met filtering criteria.")
            else: st.error("Ground truth generation failed or produced no results.")
        else: st.error("Please upload the raw data file.")
