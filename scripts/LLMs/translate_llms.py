import codecs
import torch
from transformers import pipeline, AutoModel, AutoModelForCausalLM, AutoTokenizer
import sys
import os
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from utils_llms import (
    OpenWeightTranslator,
    OpenAITranslator,
    )
import logging
import tyro
import random

load_dotenv()
# define HF_HOME
os.environ["HF_HOME"] = "SET-YOUR-HOME-DIRECTORY"                                           # SET YOUR HOME DIRECTORY

# set fixed random seed to ensure reproducible results
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

def get_translator(model_name_or_path: str):
    """Instantiate a translator for commercial systems."""

    if "gpt" in model_name_or_path:
        return OpenAITranslator(model_name_or_path)
    else:
        return OpenWeightTranslator(model_name_or_path)


def get_prompt_template(template_id: str):
    if template_id == "prompt1_NL":
        return "Can you translate the following sentence into Dutch: {sentence}"
    elif template_id == "prompt1_DE":
        return "Can you translate the following sentence into German: {sentence}"
    elif template_id == "prompt1_EL":
        return "Can you translate the following sentence into Greek: {sentence}"
    elif template_id == "prompt1_ES":
        return "Can you translate the following sentence into Spanish: {sentence}"
    elif template_id == "prompt2_NL":
        return "Can you translate the following sentences into Dutch providing all the possible alternatives in terms of gender: {sentence}"
    elif template_id == "prompt2_DE":
        return "Can you translate the following sentences into German providing all the possible alternatives in terms of gender: {sentence}"
    elif template_id == "prompt2_EL":
        return "Can you translate the following sentences into Greek providing all the possible alternatives in terms of gender: {sentence}"
    elif template_id == "prompt2_ES":
        return "Can you translate the following sentences into Spanish providing all the possible alternatives in terms of gender: {sentence}"
    else:
        raise NotImplementedError(f"Template {template_id} not implemented")


# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main(
    dataset_file: str,
    model_name_or_path: str,
    output_file: str,
    dry_run: bool = False,
    prompt_template: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
):
    # Input validation
    if not output_file or output_file.strip() == "":
        raise ValueError("output_file cannot be empty")
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    print(f"Using prompt template: {prompt_template}")
    print(f"Loading input data from: {dataset_file}")
    
    try:
        df = pd.read_csv(dataset_file, sep="\t")
        logger.info(f"Successfully loaded dataset with {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    if dry_run:
        df = df.head(10)
        logger.info("Running in dry-run mode with 10 samples")

    # Check if required column exists
    if 'source_sentences' not in df.columns:
        raise ValueError(f"Column 'source_sentences' not found in dataset. Available columns: {df.columns.tolist()}")

    input_texts = df.apply(
        lambda row: f"{row['source_sentences']}",
        axis=1,
    )
    char_count = input_texts.apply(len).sum()

    print("Applying prompt template...")

    if prompt_template is not None:  # format using the prompt template
        logger.info(f"Using prompt template: {prompt_template}")
        template_formatter = get_prompt_template(prompt_template)
        input_texts = [
            template_formatter.format(sentence=p) for p in input_texts.tolist()
        ]

    logger.info(f"Loaded {len(df)} rows with {char_count} characters.")
    logger.info(
        f"Average words per passage: {np.mean([len(t.split(' ')) for t in input_texts])}"
    )
    logger.info("Some input texts...")
    logger.info(input_texts[:3])

    ###############
    # TRANSLATION
    ###############
    try:
        translator = get_translator(model_name_or_path)
        logger.info(f"Instantiated translator: {translator}")
    except Exception as e:
        logger.error(f"Failed to instantiate translator: {e}")
        raise

    kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    try:
        completions = translator.translate(input_texts, **kwargs)
        logger.info(f"Translation completed. Generated {len(completions)} translations.")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise

    ##############################
    # POSTPROCESSING AND SAVING
    ##############################
    
    # Simple language-independent post processing to clean up translations
    def clean_translation(text):
        """Clean up the translation text by splitting on colon and keeping everything after it."""
        if not text:
            return ""
        
        # Split on colon and keep everything after it
        if ':' in text:
            text = text.split(':', 1)[1].strip()

        text = text.strip().strip('"').strip("'").strip()
        text = ' '.join(text.split())
        
        return text

    # Apply cleaning to all completions
    completions = [clean_translation(c) for c in completions]


    # Save completions (translations) to dataframe column "translation "
    df["translation"] = completions
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Save the dataframe
    logger.info(f"Saving results to: {output_file}")
    df.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Successfully saved {len(df)} rows to {output_file}")


if __name__ == "__main__":
    tyro.cli(main)
