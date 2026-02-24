"""
This script provides a command-line interface to generate answers from Vision Language Models
for a given set of questions, which can be paired with images.

It supports two modes of operation:
1.  'independent': Processes each question concurrently and independently.
2.  'feedback': Processes questions sequentially for each 'case-id', providing the previous questions
    and answers of the same case as context for the next generation.

Configuration is managed via a .env file in the root of the project, which should contain API keys and file paths.
"""

import os
import asyncio
import argparse
import base64
import logging
import pandas as pd
from pathlib import Path
import tempfile
from typing import List, Dict, Union, Tuple, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import numpy as np
from tqdm import tqdm

# -------------------------------------------------
# SECTION 1: SHARED CONFIGURATION & UTILITIES
# -------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the project root
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# File paths
INPUT_PATH = os.getenv("INPUT_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
IMAGE_DIR = os.getenv("IMAGE_DIR")

# Model Configuration
MODELS = {
    "openai": [
        {"name": "gpt-5-2025-08-07", "base_url": None, "api_key": OPENAI_API_KEY},
    ],
    "gemini": [
        {
            "name": "gemini-2.5-pro",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": GEMINI_API_KEY,
        }
    ],
}


def load_dataframe() -> pd.DataFrame:
    """
    Reads the input CSV file. If an output file already exists, it loads it to continue
    from a previous state. Otherwise, it initializes a new DataFrame with empty columns for the responses.
    """
    if OUTPUT_PATH and os.path.exists(OUTPUT_PATH):
        logger.info(
            f"Found existing output file at {OUTPUT_PATH}, continuing from previous state."
        )
        df = pd.read_csv(OUTPUT_PATH)
    elif INPUT_PATH and os.path.exists(INPUT_PATH):
        logger.info(f"Creating new output file based on {INPUT_PATH}.")
        df = pd.read_csv(INPUT_PATH)
    else:
        raise ValueError(
            "INPUT_PATH environment variable must be set and file must exist if OUTPUT_PATH is not found."
        )

    # Initialize response columns for each model if they don't exist
    for vendor, models_list in MODELS.items():
        for m in models_list:
            col = f"response_{vendor}_{m['name']}"
            if col not in df.columns:
                df[col] = pd.NA
    return df


def save_dataframe_atomic(df: pd.DataFrame):
    """
    Saves the DataFrame to the output CSV file atomically to prevent data corruption.
    It writes to a temporary file first and then replaces the original file.
    """
    if not OUTPUT_PATH:
        raise ValueError("OUTPUT_PATH environment variable must be set for saving.")

    output_path = Path(OUTPUT_PATH)
    temp_dir = output_path.parent

    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".tmp", delete=False) as tmp:
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


def encode_image(path: Union[str, Path]) -> str:
    """Encodes an image file to a base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_value_empty(value: Any) -> bool:
    """
    Determines if a given value from a DataFrame is considered 'empty'.
    Handles NaN, None, empty strings, and empty lists/tuples.
    """
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0:
        return True
    return False


def get_client(model_info: dict, use_async: bool = False) -> Union[OpenAI, AsyncOpenAI]:
    """
    Creates an OpenAI client, either synchronous or asynchronous.
    It can be configured with a custom base_url for different providers.
    """
    kwargs = {"api_key": model_info["api_key"]}
    if model_info["base_url"]:
        kwargs["base_url"] = model_info["base_url"]

    return AsyncOpenAI(**kwargs) if use_async else OpenAI(**kwargs)


def get_image_paths_for_row(row: pd.Series) -> List[Path]:
    if not IMAGE_DIR:
        return []

    roi_files: List[str] = []
    for col in ("overviews", "rois"):
        val = row.get(col, None)
        if isinstance(val, str) and val.strip():
            roi_files.extend([p.strip() for p in val.split(",") if p.strip()])

    base = Path(IMAGE_DIR)
    seen: set[str] = set()
    image_paths: List[Path] = []
    for roi in roi_files:
        if roi in seen:
            continue
        seen.add(roi)
        path = base / roi
        if path.exists():
            image_paths.append(path)
        else:
            logger.warning(f"Image file not found: {path}")
    return image_paths


# -------------------------------------------------
# SECTION 2: API CALL AND TASK PROCESSING
# -------------------------------------------------


# @retry(
#     reraise=True,
#     stop=stop_after_attempt(5),
#     wait=wait_exponential(multiplier=1, min=2, max=30),
#     retry=retry_if_exception_type(Exception),  # Consider more specific exceptions
# )
async def call_api_with_retry(
    client: Union[AsyncOpenAI, OpenAI],
    model_name: str,
    messages: List[Dict[str, Any]],
    use_async: bool,
) -> str:
    """
    Calls the chat completion API with retry logic, supporting both sync and async clients.
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": 16384,
    }

    if use_async:
        response = await client.chat.completions.create(**payload)
    else:
        response = client.chat.completions.create(**payload)

    return response.choices[0].message.content


async def process_independent_task(
    idx: int,
    row: pd.Series,
    model_info: Dict[str, Any],
    column_name: str,
) -> Tuple[int, str, str]:
    """
    Processes a single question independently. It creates a client, prepares the data,
    calls the API, and returns the result for one cell.
    """
    client = get_client(model_info, use_async=True)
    try:
        image_paths = get_image_paths_for_row(row)
        images_encoded = [encode_image(p) for p in image_paths]

        content_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": f"{row['preamble']}\n{row['question']}"}
        ]
        for img_b64 in images_encoded:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )

        messages = [{"role": "user", "content": content_parts}]

        answer = await call_api_with_retry(
            client, model_info["name"], messages, use_async=True
        )
        return idx, column_name, answer
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(
            f"Failed task at idx {idx}, model {model_info['name']}: {error_msg}"
        )
        return idx, column_name, error_msg
    finally:
        if isinstance(client, AsyncOpenAI):
            await client.close()


async def process_feedback_task(
    case_df: pd.DataFrame,
    model_info: Dict[str, Any],
    full_df_ref: pd.DataFrame,
) -> List[Tuple[int, str, str]]:
    """
    Processes all questions for a single case for a single model, maintaining chat history.
    This function is async to allow different cases to be processed concurrently, but
    the questions *within* a case are processed sequentially.
    """
    model_name = model_info["name"]
    vendor = model_info["vendor"]
    case_id = case_df["case-id"].iloc[0]
    client = get_client(
        model_info, use_async=False
    )  # Use sync client for sequential internal logic

    chat_history: List[Dict[str, Any]] = []
    results = []

    for idx, row in case_df.iterrows():
        col_name = f"response_{vendor}_{model_name}"

        # Prepare the user message for the current step
        image_paths = get_image_paths_for_row(row)
        images_encoded = [encode_image(p) for p in image_paths]

        # Combine preamble and question
        preamble = str(row.get("preamble", ""))
        question = str(row["question"])
        user_text = f"{preamble}\n{question}" if preamble else question

        # Add special instruction for the first message in a case
        if not chat_history:
            user_text = f"The following question(s) and image(s) are related to the same medical case. \n{user_text}"

        user_content_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img_b64 in images_encoded:
            user_content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )

        # If the cell is already filled, reconstruct history and skip to the next row
        if not is_value_empty(full_df_ref.loc[idx, col_name]):
            chat_history.append({"role": "user", "content": user_content_parts})
            chat_history.append(
                {"role": "assistant", "content": str(full_df_ref.loc[idx, col_name])}
            )
            continue

        # Construct messages for the API call
        messages_for_api = list(chat_history)
        messages_for_api.append({"role": "user", "content": user_content_parts})

        try:
            # The API call itself is blocking here, which is intended for sequential processing within a case
            answer = await asyncio.to_thread(
                call_api_with_retry,
                client,
                model_name,
                messages_for_api,
                use_async=False,
            )

            results.append((idx, col_name, answer))
            chat_history.append({"role": "user", "content": user_content_parts})
            chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_msg = (
                f"API Error: Case {case_id}, Model {model_name}, Row {idx}: {str(e)}"
            )
            logger.error(f"  {error_msg}")
            results.append((idx, col_name, error_msg))
            chat_history.append({"role": "user", "content": user_content_parts})
            chat_history.append({"role": "assistant", "content": error_msg})

    if hasattr(client, "close"):
        client.close()
    return results


# -------------------------------------------------
# SECTION 3: MAIN EXECUTION BLOCK & CLI
# -------------------------------------------------


async def main_async():
    """Main async function to orchestrate the entire process."""
    parser = argparse.ArgumentParser(
        description="Generate VLM answers for questions with optional image context.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["independent", "feedback"],
        help=(
            "Processing mode:\n"
            "'independent' - Process each question concurrently and independently (fast).\n"
            "'feedback' - Process questions sequentially per case, providing previous Q&A as context."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent tasks to run simultaneously (default: 20)",
    )
    args = parser.parse_args()

    if not all([INPUT_PATH, OUTPUT_PATH, IMAGE_DIR]):
        raise ValueError(
            "Please set INPUT_PATH, OUTPUT_PATH, and IMAGE_DIR in the .env file."
        )

    df = load_dataframe()
    all_tasks = []

    if args.mode == "independent":
        logger.info("Preparing tasks for 'independent' mode...")
        for idx, row in df.iterrows():
            if not isinstance(idx, int):
                logger.warning(f"Skipping row with non-integer index {idx}.")
                continue
            for vendor, models in MODELS.items():
                for m in models:
                    col = f"response_{vendor}_{m['name']}"
                    if col in df.columns and is_value_empty(row.get(col)):
                        all_tasks.append(process_independent_task(idx, row, m, col))

    elif args.mode == "feedback":
        logger.info("Preparing tasks for 'feedback' mode...")
        if "case-id" not in df.columns:
            raise ValueError("'case-id' column is required for 'feedback' mode.")

        grouped_by_case = df.groupby("case-id", sort=False)
        for case_id, case_df in grouped_by_case:
            for vendor, models in MODELS.items():
                for m in models:
                    # Add vendor to model info dict
                    model_info = m.copy()
                    model_info["vendor"] = vendor
                    # A task is to process a whole case for one model
                    all_tasks.append(process_feedback_task(case_df, model_info, df))

    if not all_tasks:
        logger.info("No pending tasks found. All responses are already complete.")
        return

    logger.info(
        f"Found {len(all_tasks)} tasks to process with max concurrency of {args.concurrency}."
    )

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_with_semaphore(task):
        async with semaphore:
            return await task

    # Wrap all tasks with semaphore
    semaphore_tasks = [run_with_semaphore(task) for task in all_tasks]

    # Unified execution with limited concurrency
    all_results = []
    with tqdm(total=len(semaphore_tasks), desc="Processing tasks") as pbar:
        for completed_task in asyncio.as_completed(semaphore_tasks):
            res = await completed_task
            if args.mode == "independent":
                all_results.append(res)
            elif args.mode == "feedback":
                all_results.extend(res)  # Feedback tasks return a list of results
            pbar.update(1)

    # Process results and save
    if all_results:
        logger.info(f"Updating dataframe with {len(all_results)} new answers...")
        for idx, col, answer in all_results:
            df.at[idx, col] = answer
        save_dataframe_atomic(df)
        logger.info(f"All done. Final results saved to {OUTPUT_PATH}")


def main():
    """Synchronous entry point to run the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
