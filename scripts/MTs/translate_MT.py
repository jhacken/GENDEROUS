import os
import requests
import logging
from typing import Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import deepl
from google.cloud import translate_v2
import tyro

load_dotenv()
os.environ["HF_HOME"] = ""                                                       ## Check your directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_translator(
    model_name_or_path: str,
    auth_key: Optional[str] = None,
    google_credentials: Optional[str] = None,
):
    if model_name_or_path == "deepl":
        return deepl.Translator(auth_key)


def target_lang(language: str) -> str:
    mapping = {"NL": "nl", "DE": "de", "EL": "el", "ES": "es"}
    try:
        return mapping[language]
    except KeyError:
        raise NotImplementedError(f"{language} not implemented")


def main(
    dataset_file: str,
    model_name_or_path: str,
    output_file: str,
    target_language: Optional[str] = None,
    auth_key: Optional[str] = None,
    dry_run: bool = False,
    max_tokens: int = 512,
):
    if not output_file or output_file.strip() == "":
        raise ValueError("output_file cannot be empty")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    print(f"Target language: {target_language}")
    print(f"Loading input data from: {dataset_file}")

    df = pd.read_csv(dataset_file, sep="\t")
    if dry_run:
        df = df.head(10)

    if "source_sentences" not in df.columns:
        raise ValueError(
            f"Column 'source_sentences' not found in dataset. Available: {df.columns.tolist()}"
        )

    input_texts = df["source_sentences"].astype(str).tolist()
    logger.info(f"Loaded {len(input_texts)} rows")

    translator = get_translator(model_name_or_path, auth_key)
    target_lang_code = target_lang(target_language)

    translated_texts = []


    if model_name_or_path == "google-translate":
        def translate_text(text, target_language=target_lang_code, api_key=None):
            url = "https://translation.googleapis.com/language/translate/v2"
            params = {
                'key': api_key,
                'q': text,
                'target': target_language
            }
            response = requests.get(url, params=params)
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(response.text)  # Print error details from API
                return None
            
            data = response.json()
            
            # Ensure 'data' key exists
            if 'data' not in data or 'translations' not in data['data']:
                print("Error: Unexpected API response format")
                print(data)  # Print response for debugging
                return None
            
            translated_text = data['data']['translations'][0]['translatedText']
            
            return translated_text

        translated_text = translate_text(input_texts, target_language=target_language, api_key=auth_key)

    elif model_name_or_path == "deepl":
        for i, text in enumerate(input_texts):
            try:
                result = translator.translate_text(
                    text, target_lang=target_language.upper()
                )
                translated_text.append(result.text)
                logger.info(f"Translated {i+1}/{len(input_texts)}")
            except Exception as e:
                logger.error(f"Translation failed for row {i}: {e}")
                translated_text.append(f"[TRANSLATION_ERROR: {e}]")


    df["translation"] = translated_text

    outdir = os.path.dirname(output_file)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved {len(df)} rows to {output_file}")


if __name__ == "__main__":
    tyro.cli(main)
