# VLM Answer Generation for pathology VQA

This document outlines the two scenarios for generating answers using a Vision-Language Model (VLM) for pathology Visual Question Answering (VQA). The accompanying script, `generate_answers_sync.py`, serves as the reference implementation for these scenarios.

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

## Quick Start

1. Copy `.env.template` to `.env` and fill in your values:

   ```
   cp .env.template .env
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   # Independent mode
   python generate_answers_sync.py --mode independent

   # Feedback mode
   python generate_answers_sync.py --mode feedback
   ```

---

## Adapting for Your Model

The script is designed so that you only need to modify **one function**: `generate_answer` in `generate_answers_sync.py`.

```python
def generate_answer(messages: list[dict], model_name: str) -> str:
    """
    Call your model and return the answer as a string.

    Args:
        messages: A list of message dicts in OpenAI chat format. Each message
                  has a "role" ("user" or "assistant") and "content" (a string
                  or a list of content parts with text and base64 images).
        model_name: The model identifier string (from MODEL_NAME env var).

    Returns:
        The model's answer as a plain string.
    """
```

The default implementation uses the OpenAI SDK, which is compatible with many providers (OpenAI, Google Gemini via their OpenAI-compatible endpoint, vLLM, Ollama, etc.). If your provider supports the OpenAI chat format, you may only need to change `API_KEY`, `API_BASE_URL`, and `MODEL_NAME` in the `.env` file without modifying any code.

The rest of the script handles data loading, image encoding, chat history management, and saving results, so you should not need to change any of that.

### Message Format

The `messages` argument follows the [OpenAI chat completions format](https://platform.openai.com/docs/api-reference/chat). For a single-turn (independent) request, it looks like:

```python
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Preamble text\nQuestion text"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }
]
```

For multi-turn (feedback) requests, previous turns are included:

```python
[
    {"role": "user", "content": [...]},       # Question 1
    {"role": "assistant", "content": "..."},   # Model's answer to Q1
    {"role": "user", "content": [...]},       # Question 2 (current)
]
```

---

## How the Two Modes Work

### Independent Mode (`--mode independent`)

Handled by `run_independent()`:

1. Iterates over each row in the dataset.
2. For each unanswered question, builds a single-turn message with text and images.
3. Calls `generate_answer` and saves the result.

### Feedback Mode (`--mode feedback`)

Handled by `run_feedback()`:

1. Groups data by `case-id`.
2. For each case, iterates through its questions **sequentially**.
3. Builds a growing `chat_history` with all prior Q&A turns from the same case.
4. Calls `generate_answer` with the full history prepended to the current question.
5. Adds an introductory instruction ("The following question(s) and image(s) are related to the same medical case.") to the first question of each case.

### Resuming Interrupted Runs

Both modes support resuming. If the script is interrupted, simply run it again as it will load the existing output file and skip questions that already have answers.

## Reproducibility

The `generate_answers_async.py` script was used to generate the answers for the OpenAI and Gemini models in the DALPHIN study. `generate_answers_sync.py` is a simplified version that runs synchronously for easier debugging and adaptation to other models. The core logic for message formatting and history management is the same in both scripts.
