# db_utils.py
import json
from datetime import datetime
import streamlit as st
from elasticsearch import Elasticsearch, exceptions as es_exceptions  # Using alias for specific exceptions

EXPERIMENTS_INDEX_NAME = "elkeval_experiments_v2"


def init_db(es_client: Elasticsearch):
    """
    Initializes the Elasticsearch index for experiments if it doesn't exist.
    Args:
        es_client: The Elasticsearch client instance.
    """
    if es_client is None:
        st.error("Elasticsearch client not available for DB (experiments index) initialization.")
        return
    try:
        if not es_client.indices.exists(index=EXPERIMENTS_INDEX_NAME):
            mapping_body = {
                "mappings": {
                    "properties": {
                        "name": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "ragas_input_source_files": {"type": "text"},
                        "ragas_output_summary_file": {"type": "keyword"},
                        "avg_faithfulness": {"type": "float"},
                        "avg_answer_relevancy": {"type": "float"},
                        "avg_llm_context_precision_with_reference": {"type": "float"},
                        "avg_llm_context_recall": {"type": "float"},
                        "retrieval_methods_used": {"type": "keyword"},
                        "notes": {"type": "text"}
                    }
                }
            }
            es_client.indices.create(index=EXPERIMENTS_INDEX_NAME, body=mapping_body)
            st.info(f"Elasticsearch index '{EXPERIMENTS_INDEX_NAME}' created for experiments.")
    except es_exceptions.TransportError as e:
        st.error(f"Elasticsearch error during experiments index initialization: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during DB (ES experiments index) initialization: {e}")


def check_experiment_name_exists(es_client: Elasticsearch, name: str) -> bool:
    """Checks if an experiment with the given name already exists."""
    if es_client is None: return False
    try:
        query = {"query": {"term": {"name": name}}}
        res = es_client.search(index=EXPERIMENTS_INDEX_NAME, body=query)
        return res['hits']['total']['value'] > 0
    except es_exceptions.NotFoundError:
        return False
    except es_exceptions.TransportError as e:
        st.warning(f"Error checking experiment name '{name}': {e}. Assuming name does not exist to allow save attempt.")
        return False
    except Exception as e:
        st.warning(f"Unexpected error checking experiment name '{name}': {e}. Assuming name does not exist.")
        return False


def save_experiment(es_client: Elasticsearch, name: str, ragas_input_files: list,
                    ragas_output_file: str, metrics: dict, retrieval_methods: list, notes: str = ""):
    """Saves a new experiment to Elasticsearch."""
    if es_client is None:
        st.error("Elasticsearch client not available. Cannot save experiment.")
        return None

    if check_experiment_name_exists(es_client, name):
        st.error(f"Experiment name '{name}' already exists. Please choose a unique name.")
        return None

    timestamp = datetime.now().isoformat()

    # The 'metrics' dict comes with keys like 'faithfulness', 'answer_relevancy', etc. (NO 'avg_' prefix)
    # The ES document fields are 'avg_faithfulness', 'avg_answer_relevancy', etc.
    document = {
        "name": name,
        "timestamp": timestamp,
        "ragas_input_source_files": ragas_input_files,
        "ragas_output_summary_file": ragas_output_file,
        # Correctly get values from the metrics dict using keys WITHOUT 'avg_'
        "avg_faithfulness": metrics.get('faithfulness'),
        "avg_answer_relevancy": metrics.get('answer_relevancy'),
        "avg_llm_context_precision_with_reference": metrics.get('llm_context_precision_with_reference'),
        "avg_llm_context_recall": metrics.get('llm_context_recall'),
        "retrieval_methods_used": list(set(retrieval_methods)),
        "notes": notes
    }

    try:
        response = es_client.index(index=EXPERIMENTS_INDEX_NAME, document=document, refresh="wait_for")
        if response.get('result') == 'created':
            st.success(f"Experiment '{name}' saved successfully to Elasticsearch with ID: {response['_id']}")
            return response['_id']
        else:
            st.error(f"Failed to save experiment '{name}' to Elasticsearch. Response: {response}")
            return None
    except es_exceptions.TransportError as e:
        st.error(f"Elasticsearch error saving experiment '{name}': {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred saving experiment '{name}': {e}")
        return None


def load_all_experiments(es_client: Elasticsearch):
    """Loads all experiments from Elasticsearch."""
    if es_client is None:
        st.error("Elasticsearch client not available. Cannot load experiments.")
        return []
    try:
        res = es_client.search(index=EXPERIMENTS_INDEX_NAME, body={"query": {"match_all": {}}}, size=1000,
                               sort=[{"timestamp": {"order": "desc"}}])
        experiments = []
        for hit in res['hits']['hits']:
            exp_data = hit['_source']
            exp_data['id'] = hit['_id']
            experiments.append(exp_data)
        return experiments
    except es_exceptions.NotFoundError:
        st.info(f"Experiments index '{EXPERIMENTS_INDEX_NAME}' not found. No experiments to load.")
        return []
    except es_exceptions.TransportError as e:
        st.error(f"Elasticsearch error loading experiments: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred loading experiments: {e}")
        return []


def delete_experiment(es_client: Elasticsearch, experiment_id: str):
    """Deletes an experiment from Elasticsearch by its _id."""
    if es_client is None:
        st.error("Elasticsearch client not available. Cannot delete experiment.")
        return False
    try:
        response = es_client.delete(index=EXPERIMENTS_INDEX_NAME, id=experiment_id, refresh="wait_for")
        return response.get('result') == 'deleted'
    except es_exceptions.NotFoundError:
        st.error(f"Experiment with ID '{experiment_id}' not found for deletion.")
        return False
    except es_exceptions.TransportError as e:
        st.error(f"Elasticsearch error deleting experiment '{experiment_id}': {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred deleting experiment '{experiment_id}': {e}")
        return False


def get_experiment_details(es_client: Elasticsearch, experiment_id: str):
    """Loads details for a single experiment by its Elasticsearch _id."""
    if es_client is None:
        st.error("Elasticsearch client not available. Cannot get experiment details.")
        return None
    try:
        response = es_client.get(index=EXPERIMENTS_INDEX_NAME, id=experiment_id)
        if response.get('found'):
            exp_data = response['_source']
            exp_data['id'] = response['_id']
            return exp_data
        return None
    except es_exceptions.NotFoundError:
        st.info(f"Experiment with ID '{experiment_id}' not found.")
        return None
    except es_exceptions.TransportError as e:
        st.error(f"Elasticsearch error loading experiment details for ID '{experiment_id}': {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading experiment details for ID '{experiment_id}': {e}")
        return None
