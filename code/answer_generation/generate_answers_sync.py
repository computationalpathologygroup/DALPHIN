"""
Synchronous CLI for generating VLM answers for pathology VQA.

This script supports two modes:
1. 'independent': Each question is processed as a standalone query.
2. 'feedback': Questions within a case are processed sequentially, with
   previous Q&A pairs provided as conversational context.

Configuration is managed via a .env file (see .env.template).

## How to adapt for your model
Modify the `generate_answer` function below. The rest of the script handles
data loading, image encoding, and orchestration — you should not need to
change anything else.
"""

import os
import argparse
import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")  # Optional, for non-OpenAI providers
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

INPUT_PATH = os.getenv("INPUT_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
IMAGE_DIR = os.getenv("IMAGE_DIR")


# -------------------------------------------------
# ADAPTER — MODIFY THIS FUNCTION FOR YOUR MODEL
# -------------------------------------------------


def generate_answer(messages: List[Dict[str, Any]], model_name: str) -> str:
    """
    Call your model and return the answer as a string.

    This is the only function you need to modify. The default implementation
    uses the OpenAI SDK, which is compatible with many providers (OpenAI,
    Google Gemini via OpenAI-compatible endpoint, vLLM, Ollama, etc.).

    Args:
        messages: A list of message dicts in OpenAI chat format. Each message
                  has a "role" ("user" or "assistant") and "content" (a string
                  or a list of content parts with text and base64 images).
                  Example for a single-turn request:
                  [
                      {
                          "role": "user",
                          "content": [
                              {"type": "text", "text": "What is shown?"},
                              {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                          ]
                      }
                  ]
        model_name: The model identifier string (from MODEL_NAME env var).

    Returns:
        The model's answer as a plain string.
    """
    kwargs = {"api_key": API_KEY}
    if API_BASE_URL:
        kwargs["base_url"] = API_BASE_URL

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=16384,
    )
    return response.choices[0].message.content


# -------------------------------------------------
# UTILITIES — NO CHANGES NEEDED BELOW THIS LINE
# -------------------------------------------------


def load_dataframe(response_column: str) -> pd.DataFrame:
    """Load the input/output CSV and ensure the response column exists."""
    if OUTPUT_PATH and os.path.exists(OUTPUT_PATH):
        logger.info(f"Resuming from existing output file: {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    elif INPUT_PATH and os.path.exists(INPUT_PATH):
        logger.info(f"Starting fresh from input file: {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH)
    else:
        raise ValueError(
            "INPUT_PATH must be set and the file must exist if OUTPUT_PATH is not found."
        )

    if response_column not in df.columns:
        df[response_column] = pd.NA
    return df


def save_dataframe_atomic(df: pd.DataFrame):
    """Save the DataFrame atomically to prevent corruption on interruption."""
    if not OUTPUT_PATH:
        raise ValueError("OUTPUT_PATH must be set.")

    output_path = Path(OUTPUT_PATH)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".tmp", delete=False
    ) as tmp:
        temp_path = Path(tmp.name)

    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, output_path)
    except Exception as e:
        logger.error(f"Error saving DataFrame: {e}")
        if temp_path.exists():
            os.remove(temp_path)
    finally:
        if temp_path.exists():
            os.remove(temp_path)


def encode_image(path: Path) -> str:
    """Encode an image file to a base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_value_empty(value) -> bool:
    """Check if a DataFrame cell value is empty (NaN, None, or blank string)."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def get_image_paths_for_row(row: pd.Series) -> List[Path]:
    """Collect and deduplicate image paths for a row from 'thumbnail' and 'rois' columns."""
    if not IMAGE_DIR:
        return []

    filenames: List[str] = []
    for col in ("thumbnail", "rois"):
        val = row.get(col, None)
        if isinstance(val, str) and val.strip():
            filenames.extend([p.strip() for p in val.split(",") if p.strip()])

    base = Path(IMAGE_DIR)
    seen: set = set()
    paths: List[Path] = []
    for name in filenames:
        if name in seen:
            continue
        seen.add(name)
        path = base / name
        if path.exists():
            paths.append(path)
        else:
            logger.warning(f"Image file not found: {path}")
    return paths


def build_user_content(row: pd.Series) -> List[Dict[str, Any]]:
    """Build the content parts list (text + images) for a single question."""
    preamble = str(row.get("preamble", ""))
    question = str(row["question"])
    text = f"{preamble}\n{question}" if preamble else question

    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]

    for path in get_image_paths_for_row(row):
        img_b64 = encode_image(path)
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        )

    return parts


# -------------------------------------------------
# MODE IMPLEMENTATIONS
# -------------------------------------------------


def run_independent(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Process each question independently (no conversational context)."""
    pending = df[df[col].apply(is_value_empty)].index.tolist()
    logger.info(f"Independent mode: {len(pending)} questions to process.")

    for idx in tqdm(pending, desc="Independent"):
        row = df.loc[idx]
        content = build_user_content(row)
        messages = [{"role": "user", "content": content}]

        try:
            answer = generate_answer(messages, MODEL_NAME)
            df.at[idx, col] = answer
        except Exception as e:
            logger.error(f"Error at row {idx}: {e}")
            df.at[idx, col] = f"Error: {e}"

        save_dataframe_atomic(df)

    return df


def run_feedback(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Process questions sequentially per case, providing prior Q&A as context."""
    if "case-id" not in df.columns:
        raise ValueError("'case-id' column is required for feedback mode.")

    cases = df.groupby("case-id", sort=False)
    logger.info(f"Feedback mode: {len(cases)} cases to process.")

    for case_id, case_df in tqdm(cases, desc="Cases"):
        chat_history: List[Dict[str, Any]] = []

        for idx, row in case_df.iterrows():
            content = build_user_content(row)

            # Add case-level instruction to the first message
            if not chat_history:
                content[0]["text"] = (
                    "The following question(s) and image(s) are related to the same "
                    f"medical case.\n{content[0]['text']}"
                )

            # If already answered, reconstruct history and skip
            if not is_value_empty(df.loc[idx, col]):
                chat_history.append({"role": "user", "content": content})
                chat_history.append(
                    {"role": "assistant", "content": str(df.loc[idx, col])}
                )
                continue

            messages = chat_history + [{"role": "user", "content": content}]

            try:
                answer = generate_answer(messages, MODEL_NAME)
                df.at[idx, col] = answer
            except Exception as e:
                answer = f"Error: {e}"
                logger.error(f"Error at case {case_id}, row {idx}: {e}")
                df.at[idx, col] = answer

            chat_history.append({"role": "user", "content": content})
            chat_history.append({"role": "assistant", "content": answer})

            save_dataframe_atomic(df)

    return df


# -------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM answers for pathology VQA.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["independent", "feedback"],
        help=(
            "Processing mode:\n"
            "  'independent' — each question is answered in isolation.\n"
            "  'feedback'    — questions within a case share conversational context."
        ),
    )
    args = parser.parse_args()

    if not all([INPUT_PATH, OUTPUT_PATH, IMAGE_DIR]):
        raise ValueError("Set INPUT_PATH, OUTPUT_PATH, and IMAGE_DIR in your .env file.")

    response_col = f"response_{MODEL_NAME}"
    df = load_dataframe(response_col)

    if args.mode == "independent":
        run_independent(df, response_col)
    else:
        run_feedback(df, response_col)

    logger.info(f"Done. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
