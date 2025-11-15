from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
import argparse
import logging
import os
import json
import re
import csv
from typing import Tuple, Optional

import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import list_datasets

from logger_config import init_logger
from utils import parse_answer, clean_index_columns


parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompts_path",
    type=str,
    required=True,
    help="HF path or local CSV (via --run_local) for prompts"
)
parser.add_argument(
    "--answers_path",
    type=str,
    required=True,
    help="HF path or local CSV (via --run_local) to save answers"
)
parser.add_argument(
    "--model_path",
    type=str,
    nargs="+",
    required=True,
    help="API model identifiers (e.g., 'gemini-2.5-flash')"
)
parser.add_argument(
    "--run_local",
    action="store_true",
    help="Read and write CSVs under eval_processing instead of pushing to Hub"
)
parser.add_argument(
    "--use_outlines",
    action="store_true",
    help="If set, require structured JSON via outlines. Not implemented for API."
)
parser.add_argument(
    "--answers_shard_suffix",
    type=str,
    default="",
    help="Optional suffix for local CSV filename (kept for symmetry)."
)

args = parser.parse_args()
init_logger()

SYSTEM_RE = re.compile(r"<system>\s*(.*?)\s*</system>", re.DOTALL | re.IGNORECASE)
USER_RE   = re.compile(r"<user>\s*(.*?)\s*</user>",     re.DOTALL | re.IGNORECASE)

def extract_system_user(prompt: str) -> Tuple[str, str]:
    m_sys = SYSTEM_RE.search(prompt or "")
    m_usr = USER_RE.search(prompt or "")
    system = m_sys.group(1).strip() if m_sys else ""
    user   = m_usr.group(1).strip() if m_usr else ""
    return system, user


def get_prompt_for_model(prompt_json_str: str, model_path: str) -> str:
    try:
        prompt_data = json.loads(prompt_json_str)
        for _, info in prompt_data.items():
            if model_path in info["models"]:
                return info["prompt"]
        raise KeyError(f"Model {model_path} not found in prompt data")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in prompt cell: {e}")


def clean_data_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    def clean_text(text):
        if pd.isna(text):
            return text
        text = str(text)
        text = re.sub(r'\\([uU][0-9a-fA-F]{4})', r'\\\\\1', text)
        text = re.sub(r'\\([^uU])', r'\\\\\1', text)
        return text
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(clean_text)
    return df

def ensure_string_columns(ds: Dataset) -> Dataset:
    target = ['model_answer','parsed_model_answer','label',
              'prompt','shot_indices','benchmark','model_name','id_bench','explanation']
    for col in target:
        if col in ds.column_names:
            ds = ds.cast_column(col, "string")
    return ds

_genai_client = genai.Client()

def generate_with_gemini(model: str, prompt_str: str, timeout_s: Optional[float] = None) -> str:
    system, user = extract_system_user(prompt_str)
    cfg = GenerateContentConfig(system_instruction=[system] if system else None)
    kwargs = dict(model=model, contents=user, config=cfg)
    if timeout_s is not None:
        kwargs["http_options"] = HttpOptions(timeout=timeout_s)
    resp = _genai_client.models.generate_content(**kwargs)
    return getattr(resp, "text", "")

def generate_with_gemini_outlines(*_args, **_kwargs) -> str:
    raise NotImplementedError("Outlines not implemented for Gemini API yet.")


def main():
    if args.use_outlines:
        logging.warning("Outlines is not supported for API models. Proceeding with normal execution.")

    if args.run_local:
        os.makedirs("eval_processing", exist_ok=True)
        fn = args.prompts_path.replace("/", "_").replace("-", "_") + ".csv"
        fp = os.path.join("eval_processing", fn)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Prompts file not found: {fp}")
        df = pd.read_csv(fp)
        df = clean_index_columns(df)
        dataset = Dataset.from_pandas(df)
        logging.info(f"Loaded prompts locally: {fp}")
    else:
        dataset = load_dataset(args.prompts_path, split="train")
        logging.info(f"Loaded prompts from Hub: {args.prompts_path}")

    all_outputs = []

    for model_path in args.model_path:
        logging.info(f"** RUNNING API MODEL: {model_path}")
        model_name = model_path.split("/")[-1]

        ds_m = dataset.map(lambda ex: {"model_name": model_name})

        def map_one(example):
            ptxt = get_prompt_for_model(example["prompt"], model_path)
            try:
                ans = generate_with_gemini(model_path, ptxt, timeout_s=None)
            except Exception as e:
                ans = f"__API_ERROR__: {e}"

            example["prompt"] = ptxt
            example["model_answer"] = ans
            example["explanation"] = None
            parsed = parse_answer({"benchmark": example["benchmark"], "model_answer": ans})
            example["parsed_model_answer"] = parsed
            return example

        ds_m = ds_m.map(map_one, desc=f"{model_name} (api)")
        ds_m = ensure_string_columns(ds_m)
        all_outputs.append(ds_m)

    out = all_outputs[0] if len(all_outputs) == 1 else Dataset.from_pandas(
        pd.concat([d.to_pandas() for d in all_outputs], ignore_index=True)
    )

    out = out.map(lambda ex, i: {**ex, "id": i + 1}, with_indices=True)

    api_model_names = set([mp.split("/")[-1] for mp in args.model_path])

    if args.run_local:
        base = args.answers_path.replace("/", "_").replace("-", "_")
        suffix = args.answers_shard_suffix or ""
        answers_fp = os.path.join("eval_processing", f"{base}{suffix}.csv")

        if os.path.exists(answers_fp):
            prev = pd.read_csv(answers_fp)
            prev = clean_index_columns(prev)
            prev = prev[~prev.get("model_name", "").isin(api_model_names)]
            merged_df = pd.concat([prev, out.to_pandas()], ignore_index=True)
        else:
            merged_df = out.to_pandas()

        merged_df = clean_data_for_csv(merged_df)
        merged_df.to_csv(
            answers_fp,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            doublequote=True,
            escapechar='\\',
            lineterminator='\n'
        )
        logging.info(f"**SAVED API MODELS AT: {answers_fp}")
    else:
        try:
            possible = list_datasets(search=args.answers_path)
            exists = any(ds.id == args.answers_path for ds in possible)
        except Exception:
            exists = False

        if exists:
            for attempt in range(3):
                try:
                    original = load_dataset(args.answers_path, split="train")
                    break
                except Exception as e:
                    if attempt == 2:
                        raise Exception(f"Failed to load existing dataset after 3 attempts: {e}")
                    logging.info(f"Load attempt {attempt+1} failed, retrying...")
            ex_df = original.to_pandas()
            ex_df = ex_df[~ex_df.get("model_name", "").isin(api_model_names)]
            merged_df = pd.concat([ex_df, out.to_pandas()], ignore_index=True)
        else:
            merged_df = out.to_pandas()

        Dataset.from_pandas(merged_df).push_to_hub(args.answers_path)
        logging.info(f"**PUSHED API MODELS TO: {args.answers_path}")

if __name__ == "__main__":
    main()
