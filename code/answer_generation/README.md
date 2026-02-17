# VLM Answer Generation for pathology VQA

This document outlines the two scenarios for generating answers using a Vision-Language Model (VLM) for pathology Visual Question Answering (VQA). The accompanying script, `generate_answers_cli.py`, serves as a reference implementation for these scenarios.

## General Premise

The goal is to benchmark VLMs on pathology VQA tasks under two distinct conditions.

### Scenario 1: Independent Generation

In this scenario, each question is treated as a standalone query. The model receives the question text and any associated histopathology images without any context from previous interactions. This mode is designed to test the model's ability to answer questions based solely on the information provided in a single prompt.

- **Input:** A question and one or more related images.
- **Process:** The model processes the input and generates an answer.
- **State:** No memory or state is maintained between questions, even if they belong to the same medical case.

### Scenario 2: Contextual (Feedback) Generation

This scenario simulates a more realistic diagnostic workflow where a pathologist might ask a series of follow-up questions about the same medical case. The model maintains a conversational history throughout a single case.

- **Input:** A question, its related images, and the full history of preceding questions and their corresponding *model-generated answers* from the *same case*.
- **Process:** The model uses the provided context to inform its answer for the current question.
- **State:** The conversation history is built up sequentially for each case. This history is reset at the beginning of every new case. Note: the expected (ground truth) answers are not included in the context, only the model's previous answers.

---

## Implementation Guide for New Models

The script `generate_answers_cli.py` provides a template for implementing both scenarios. Your team should adapt the core logic to fit your model's specific API and requirements. The key entry point is the `--mode` flag, which switches between `independent` and `feedback`.
Any environment variables should be set in your shell or in the `.env` file. You can refer to `.env.template` for the required variables.
Note: The script is made to asynchronously call the model API for efficiency, but the core logic can be adapted to synchronous calls if needed.

### For Independent Generation (`--mode independent`)

This logic is primarily handled by the `process_independent_task` function.

- **Reference Function:** `process_independent_task(idx, row, model_info, column_name)`
- **How it works:**
    1. For each question (a `row` in the data), this function is called.
    2. It retrieves the question text (`row['question']`) and preamble (`row['preamble']`).
    3. It finds and encodes the relevant images using `get_image_paths_for_row` and `encode_image`.
    4. It constructs a `messages` payload containing just the current question and images.
    5. It calls the model API via `call_api_with_retry`.

To adapt this, you should replicate the process of creating a single, stateless prompt for each question and sending it to your model endpoint.

### For Contextual (Feedback) Generation (`--mode feedback`)

This logic is handled by the `process_feedback_task` function, which processes all questions for a single medical case.

- **Reference Function:** `process_feedback_task(case_df, model_info, full_df_ref)`
- **How it works:**
    1. The data is grouped by `case-id`, and this function receives a dataframe (`case_df`) for one case.
    2. A `chat_history` list is initialized to store the conversation.
    3. The function iterates through the questions *sequentially* within the case.
    4. For each question, it constructs the user message with the text and images.
    5. Crucially, it prepends the entire `chat_history` to the current message before making the API call.
    6. After the model responds, both the user message and the assistant's answer are appended to the `chat_history` list, building the context for the next turn.
    7. A special instruction, "The following question(s) and image(s) are related to the same medical case," is added to the very first question of a case to prime the model.

To adapt this, you must implement a similar loop for each case that maintains a running list of the conversation turns and includes this history in each subsequent API call for that case.
