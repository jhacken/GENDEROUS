import inflect
from dataclasses import dataclass
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from typing import List, Union
from openai import OpenAI
import os
import logging
import time
import random
import http.client
from vllm import LLM, SamplingParams
import json

logger = logging.getLogger(__name__)


class OpenWeightTranslator:
    def __init__(self, model_name_or_path: str):
        self.llm = LLM(
            model=model_name_or_path,
            dtype="bfloat16",
            max_model_len=4096,
            enable_prefix_caching=True,
            disable_sliding_window=True,
        )

    def translate(
        self,
        texts: List[str],
        apply_chat_template: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        if apply_chat_template:
            texts = [
                self.llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_template=True,
                )
                for t in texts
            ]
        outputs = self.llm.generate(texts, sampling_params)
        output_texts = [o.outputs[0].text for o in outputs]
        return output_texts


class OpenAITranslator:

    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def translate(
        self,
        texts: List[str],
        apply_chat_template: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        completions = list()
        for count, prompt in tqdm(
            enumerate(texts), desc="GPT API calls", total=len(texts)
        ):
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=max_tokens,  # Use the parameter instead of hardcoded 2048
                model=self.model_name,
                temperature=temperature,
            )
            translated_text = response.choices[0].message.content
            completions.append(translated_text)
            if (count + 1) % 60 == 0:
                logger.info(
                    f"Completed {count + 1} prompts. Sleeping for one minute..."
                )
                time.sleep(65)
        return completions