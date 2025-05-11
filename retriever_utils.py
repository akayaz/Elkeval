# retriever_utils.py
import os
import json
from dotenv import load_dotenv, dotenv_values
from elasticsearch import Elasticsearch, exceptions # Import the exceptions module
from openai import AzureOpenAI
import streamlit as st

# --- Configuration and Initialization ---

def load_config():
    """Loads configuration from .env file, ensuring all necessary keys are checked."""
    load_dotenv(override=True)
    config = {}
    # Elasticsearch
    config["ES_URL"] = os.getenv("ES_URL")
    config["ES_USER"] = os.getenv("ES_USER")
    config["ES_PASSWORD"] = os.getenv("ES_PASSWORD")
    config["ES_CID"] = os.getenv("ES_CID")
    config["ES_INDEX_NAME_ENV"] = os.getenv("ES_INDEX_NAME")
    # Azure OpenAI - Main Model
    config["AZURE_OPENAI_KEY"] = os.getenv("AZURE_OPENAI_KEY")
    config["AZURE_OPENAI_BASE"] = os.getenv("AZURE_OPENAI_BASE")
    config["AZURE_OPENAI_DEPLOYMENT"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    config["AZURE_OPENAI_MODEL"] = os.getenv("AZURE_OPENAI_MODEL")
    config["AZURE_OPENAI_VERSION"] = os.getenv("AZURE_OPENAI_VERSION")
    # Azure OpenAI - Embedding Model
    config["AZURE_API_KEY_EMBEDDING"] = os.getenv("AZURE_API_KEY_EMBEDDING", config.get("AZURE_OPENAI_KEY"))
    config["AZURE_ENGINE_EMBEDDING"] = os.getenv("AZURE_ENGINE_EMBEDDING")
    config["AZURE_API_VERSION_EMBEDDING"] = os.getenv("AZURE_API_VERSION_EMBEDDING")
    config["AZURE_EMBEDDING_MODEL"] = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002")
    return config

def initialize_clients(config):
    """Initializes and returns Elasticsearch and AzureOpenAI clients."""
    es_client = None
    openai_client = None
    try: # Elasticsearch Client
        if config.get("ES_CID") and config.get("ES_USER") and config.get("ES_PASSWORD"):
            es_client = Elasticsearch(cloud_id=config.get("ES_CID"), basic_auth=(config["ES_USER"], config["ES_PASSWORD"]), request_timeout=30)
        elif config.get("ES_URL"):
            auth = (config["ES_USER"], config["ES_PASSWORD"]) if config.get("ES_USER") and config.get("ES_PASSWORD") else None
            es_client = Elasticsearch(hosts=[config.get("ES_URL")], basic_auth=auth, request_timeout=30)
        if es_client and not es_client.ping(): st.warning("Elasticsearch client initialized but ping failed.")
    except Exception as e: st.error(f"Failed to initialize Elasticsearch client: {e}"); es_client = None
    try: # Azure OpenAI Client
        if config.get("AZURE_OPENAI_KEY") and config.get("AZURE_OPENAI_BASE") and config.get("AZURE_OPENAI_VERSION"):
            openai_client = AzureOpenAI(api_key=config["AZURE_OPENAI_KEY"], api_version=config["AZURE_OPENAI_VERSION"], azure_endpoint=config["AZURE_OPENAI_BASE"])
        else: st.warning("Azure OpenAI client credentials (KEY, BASE, VERSION) not fully provided.")
    except Exception as e: st.error(f"Failed to initialize Azure OpenAI client: {e}"); openai_client = None
    return es_client, openai_client

# --- Index and Field Constants (Defaults, overridden by UI selection) ---
INDEX_NAME = "dragonball-law-index-chunk" # Default from user's file

# --- Functions for Index Mapping and Field Extraction ---
def get_index_mapping(es_client: Elasticsearch, index_name: str):
    if not es_client or not index_name: return None
    try:
        if es_client.indices.exists(index=index_name):
            mapping = es_client.indices.get_mapping(index=index_name)
            return mapping.get(index_name, {}).get("mappings", {})
        st.warning(f"Index '{index_name}' does not exist for mapping retrieval.")
        return None
    except exceptions.ElasticsearchException as e: # Use the imported 'exceptions' module
        st.error(f"Elasticsearch error fetching mapping for index '{index_name}': {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        st.error(f"An unexpected error occurred while fetching mapping for index '{index_name}': {e}")
        return None


def extract_fields_from_mapping(mapping_properties: dict, desired_types: list, current_path="", fields=None):
    if fields is None: fields = []
    if not isinstance(mapping_properties, dict): return fields
    for field_name, field_info in mapping_properties.items():
        if not isinstance(field_info, dict): continue
        full_field_path = f"{current_path}.{field_name}" if current_path else field_name
        field_type = field_info.get("type")
        if field_type in desired_types: fields.append(full_field_path)
        if "properties" in field_info: extract_fields_from_mapping(field_info["properties"], desired_types, full_field_path, fields)
    return sorted(list(set(fields)))

# --- Elasticsearch Query Functions (Adhering to User's Retriever Structure with Dynamic Fields) ---

def query_fulltext(es_client: Elasticsearch, query: str, index_name: str, field_to_query: str, size: int = 1):
    """Performs a full-text (multi_match) search on a specified text field."""
    if not es_client: return []
    if not field_to_query: st.warning("No field selected for full-text search."); return []
    try:
        es_query = {
            "retriever": {
                "standard": {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": [field_to_query]
                        }
                    }
                }
            },
            "size": size
        }
        result = es_client.search(index=index_name, body=es_query)
        return result["hits"]["hits"]
    except exceptions.ElasticsearchException as e: st.error(f"Full-text search error: {e}"); return []
    except Exception as e: st.error(f"Unexpected error in full-text search: {e}"); return []


def query_semantic(es_client: Elasticsearch, query: str, index_name: str, semantic_text_field: str, size: int = 5):
    """Performs a semantic search on a specified semantic_text_field."""
    if not es_client: return []
    if not semantic_text_field: st.warning("No field selected for semantic search."); return []
    try:
        es_query = {
            "retriever": {
                "standard": {
                    "query": {
                        "semantic": {
                            "field": semantic_text_field,
                            "query": query
                        }
                    }
                }
            },
            "highlight": {
                "fields": {
                    semantic_text_field: {
                        "type": "semantic",
                        "number_of_fragments": 2,
                        "order": "score"
                    }
                }
            },
            "size": size
        }
        result = es_client.search(index=index_name, body=es_query)
        return result["hits"]["hits"]
    except exceptions.ElasticsearchException as e: st.error(f"Semantic search error: {e}"); return []
    except Exception as e: st.error(f"Unexpected error in semantic search: {e}"); return []

def query_semantic_bge(es_client: Elasticsearch, query: str, index_name: str, semantic_text_field: str, size: int = 5):
    """Performs a semantic search, assuming semantic_text_field is configured for BGE."""
    return query_semantic(es_client, query, index_name, semantic_text_field, size)


def query_hybrid(es_client: Elasticsearch, query: str, index_name: str, field_to_query: str, semantic_text_field: str, size: int = 5):
    """Performs a hybrid search using RRF with specified text and semantic fields."""
    if not es_client: return []
    if not field_to_query: st.warning("No lexical field for hybrid search."); return []
    if not semantic_text_field: st.warning("No semantic field for hybrid search."); return []
    try:
        es_query = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "semantic": {
                                        "field": semantic_text_field,
                                        "query": query
                                    }
                                }
                            }
                        },
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": [field_to_query]
                                    }
                                }
                            }
                        }
                    ],
                    "rank_window_size": 50,
                    "rank_constant": 60
                }
            },
            "highlight": {
                "fields": {
                    semantic_text_field: {
                        "type": "semantic",
                        "number_of_fragments": 2,
                        "order": "score"
                    }
                }
            },
            "size": size
        }
        result = es_client.search(index=index_name, body=es_query)
        return result["hits"]["hits"]
    except exceptions.ElasticsearchException as e: st.error(f"Hybrid (RRF) search error: {e}"); return []
    except Exception as e: st.error(f"Unexpected error in hybrid (RRF) search: {e}"); return []

def query_hybrid_linear(es_client: Elasticsearch, query: str, index_name: str, field_to_query: str, semantic_text_field: str, size: int = 5):
    """Performs a hybrid search using linear combination."""
    if not es_client: return []
    if not field_to_query: st.warning("No lexical field for linear hybrid search."); return []
    if not semantic_text_field: st.warning("No semantic field for linear hybrid search."); return []
    try:
        es_query = {
            "retriever": {
                "linear": {
                    "retrievers": [
                        {
                            "retriever": {
                                "standard": {
                                    "query": {
                                        "semantic": {
                                            "field": semantic_text_field,
                                            "query": query
                                        }
                                    }
                                }
                            },
                            "weight": 2, "normalizer": "minmax"
                        },
                        {
                            "retriever": {
                                "standard": {
                                    "query": {
                                        "multi_match": {
                                            "query": query,
                                            "fields": [field_to_query]
                                        }
                                    }
                                }
                            },
                            "weight": 1.5, "normalizer": "minmax"
                        }
                    ],
                    "rank_window_size": 50
                }
            },
            "size": size
        }
        result = es_client.search(index=index_name, body=es_query)
        return result["hits"]["hits"]
    except exceptions.ElasticsearchException as e: st.error(f"Hybrid Linear search error: {e}"); return []
    except Exception as e: st.error(f"Unexpected error in hybrid linear search: {e}"); return []

def query_hybrid_bge(es_client: Elasticsearch, query: str, index_name: str, field_to_query: str, semantic_text_field: str, size: int = 5):
    """Performs a hybrid search using RRF with a BGE semantic field."""
    return query_hybrid(es_client, query, index_name, field_to_query, semantic_text_field, size)

def query_hybrid_linear_bge(es_client: Elasticsearch, query: str, index_name: str, field_to_query: str, semantic_text_field: str, size: int = 5):
    """Performs a hybrid search using linear combination with a BGE semantic field."""
    return query_hybrid_linear(es_client, query, index_name, field_to_query, semantic_text_field, size)


def query_hybrid_rerank(es_client: Elasticsearch, query: str, index_name: str, field_to_query: str, semantic_text_field: str, rerank_content_field: str, rank_window_size: int = 10, min_score: float = 0.8):
    """Performs a hybrid search with Cohere Rerank."""
    if not es_client: return []
    if not field_to_query or not semantic_text_field or not rerank_content_field:
        st.warning("Missing fields for hybrid rerank search."); return []
    try:
        base_rrf_retrievers = [
            {"standard": {"query": {"semantic": {"field": semantic_text_field, "query": query}}}},
            {"standard": {"query": {"multi_match": {"query": query, "fields": [field_to_query]}}}}
        ]
        es_query = {
            "retriever": {
                "text_similarity_reranker": {
                    "retriever": {"rrf": {"retrievers": base_rrf_retrievers, "rank_window_size": rank_window_size * 3, "rank_constant": 60}},
                    "inference_id": "cohere-rerank-ayimka-prod",
                    "inference_text": query, "field": rerank_content_field,
                    "rank_window_size": rank_window_size, "min_score": min_score
                }
            }
        }
        result = es_client.search(index=index_name, body=es_query)
        return result["hits"]["hits"]
    except exceptions.ElasticsearchException as e: st.error(f"Hybrid Rerank search error: {e}"); return []
    except Exception as e: st.error(f"Unexpected error in hybrid rerank search: {e}"); return []

def query_hybrid_bge_rerank(es_client: Elasticsearch, query: str, index_name: str, field_to_query: str, semantic_text_field: str, rerank_content_field: str, rank_window_size: int = 10, min_score: float = 0.8):
    """Performs a hybrid search with BGE semantic and Cohere Rerank."""
    return query_hybrid_rerank(es_client, query, index_name, field_to_query, semantic_text_field, rerank_content_field, rank_window_size, min_score)

# --- OpenAI Interaction (from user's uploaded file) ---

def create_openai_prompt(results, primary_content_field: str):
    """Creates the prompt for OpenAI using Elasticsearch results."""
    context = ""; raw_context = []
    for hit in results:
        hit_content_parts = []
        if "highlight" in hit:
            for field_name, highlights in hit["highlight"].items(): hit_content_parts.extend(highlights)
        source_data = hit.get("_source", {}); content_from_source = source_data.get(primary_content_field)
        if content_from_source:
            str_content_from_source = str(content_from_source)
            raw_context.append(str_content_from_source)
            if not hit_content_parts: hit_content_parts.append(str_content_from_source)
        elif not hit_content_parts: raw_context.append(f"Missing content for field '{primary_content_field}' in this hit.")
        if hit_content_parts: context += "\n---\n".join(map(str,hit_content_parts)) + "\n\n"
    prompt_template = """
<|system|>  
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.

Context:
{context_for_llm}
<|user|>
"""
    if not context.strip():
        final_prompt = prompt_template.replace("{context_for_llm}", "No relevant context was extracted from the search results.\nPlease state that you cannot answer based on the provided information.")
        raw_context = ["No relevant context extracted"] if not raw_context else raw_context
    else: final_prompt = prompt_template.replace("{context_for_llm}", context.strip())
    return final_prompt, raw_context

def generate_openai_completion(openai_client: AzureOpenAI, model_deployment_name: str, system_prompt: str, question: str, temperature: float = 1.0):
    """Generates a completion using the OpenAI client."""
    if not openai_client: st.error("OpenAI client not initialized."); return None
    if not model_deployment_name: st.error("Azure OpenAI deployment name not provided."); return None
    try:
        response = openai_client.chat.completions.create(
            model=model_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e: st.error(f"Error generating OpenAI completion: {e}"); return None

# --- Data Handling (from user's uploaded file) ---

def load_questions(filepath: str):
    """Loads questions from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file: data = json.load(file)
        return data
    except FileNotFoundError: st.error(f"Error: Input file not found at {filepath}"); return []
    except json.JSONDecodeError: st.error(f"Error: Could not decode JSON from {filepath}"); return []

def append_result(output_filepath: str, result_data: dict):
    """Loads existing results, appends new data, and saves back to JSON."""
    outputs = []
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding='utf-8') as f: outputs = json.load(f)
            if not isinstance(outputs, list): st.warning(f"Warning: Output file {output_filepath} did not contain a list. Overwriting."); outputs = []
        except json.JSONDecodeError: st.warning(f"Warning: Could not decode JSON from {output_filepath}. Overwriting."); outputs = []
        except Exception as e: st.error(f"Error reading {output_filepath}: {e}. Starting fresh."); outputs = []
    outputs.append(result_data)
    try:
        with open(output_filepath, "w", encoding='utf-8') as f: json.dump(outputs, f, indent=4, ensure_ascii=False)
    except Exception as e: st.error(f"Error writing to {output_filepath}: {e}")

