# prompts.py
"""
This file stores all the large prompt template strings for the RAG application.
"""

QA_GENERATION_PROMPT_TEMPLATE = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}
Output:::"""

QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT_TEMPLATE = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}
Context: {context}
Answer::: """

QUESTION_RELEVANCE_CRITIQUE_PROMPT_TEMPLATE = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to lawyer and prosecutors, attorneys and people working in the field of Law.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer::: """

QUESTION_STANDALONE_CRITIQUE_PROMPT_TEMPLATE = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself
Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer::: """

VISUALIZATION_EXPLANATION_PROMPT_TEMPLATE = """
You are an expert AI assistant specializing in analyzing and explaining Retrieval Augmented Generation (RAG) system evaluation results.
You will be provided with a summary of RAGAS evaluation metrics for one or more documents/experiments.
The metrics are typically scored between 0 and 1 (often presented as percentages, where 100% is best). Key RAGAS metrics include:

- **Faithfulness**: Measures the factual accuracy of the generated answer against the retrieved context. Higher scores mean the answer is well-supported by the context and does not contain hallucinations.
- **Answer Relevancy**: Assesses how relevant the generated answer is to the given question. It penalizes incomplete or redundant answers. Higher scores are better.
- **Context Precision (or LLM Context Precision With Reference)**: Evaluates the relevance of the retrieved context. It checks if the context chunks used for generation are focused on the question. Higher scores indicate more signal and less noise in the retrieved context.
- **Context Recall (or LLM Context Recall)**: Measures the extent to which the retrieved context covers all the necessary information from the ground truth answer. Higher scores mean the context is comprehensive enough.

The input data is a summary of average scores for these metrics, usually per document or experiment.

Your task is to:
1.  Analyze the provided scores for each document/experiment.
2.  Interpret what these scores mean for the RAG system's performance. For example, high Faithfulness and Answer Relevancy but low Context Recall might suggest the system is good at answering with what it's given, but it's not retrieving all necessary information.
3.  If multiple documents/experiments are present, briefly compare their performance, highlighting key differences.
4.  Identify clear strengths and weaknesses based on the metrics.
5.  Suggest 1-2 actionable areas for potential improvement if weaknesses are identified (e.g., if Context Precision is low, suggest improving retrieval or chunking strategies).
6.  Provide a concise, easy-to-understand explanation. Use Markdown for formatting (e.g., bolding, bullet points).

Here is the summary of RAGAS evaluation metrics:

{summary_data_string}

Please provide your analysis:
"""

TRANSLATE_PROMPT_TEMPLATE = """
Translate the following text from English to {target_language}.
Ensure the meaning and placeholders (like {{context}}, {{question}}, {{answer}}) are preserved exactly.
Only provide the translated text, with no additional explanations or preamble.

Original English text:
---
{text_to_translate}
---

Translated text:
"""
