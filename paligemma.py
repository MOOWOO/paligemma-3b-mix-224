import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from transformers import BitsAndBytesConfig
from swarms import BaseMultiModalModel
from huggingface_hub import login
import os
from dotenv import load_dotenv
import logging

from logging_config import logger  # Import the logger

torch.set_default_device("cuda")

load_dotenv()
access_token = os.getenv('HF_TOKEN')
login(access_token)

class PaliGemma(BaseMultiModalModel):
    def __init__(
        self,
        model_id: str = "google/paligemma-3b-mix-224",
        max_new_tokens: int = 50,
        skip_special_tokens: bool = True,
        *args,
        **kwargs
    ):
        logger.debug("Initializing PaliGemma model")
        self.model_id = model_id
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            low_cpu_mem_usage=True,
            *args,
            **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens
        self.skip_special_tokens = skip_special_tokens
        logger.debug("PaliGemma model initialized successfully")

    def run(self, task: str = None, image_url: str = None, *args, **kwargs):
        logger.debug(f"Running model with task: {task} and image URL: {image_url}")
        raw_image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.processor(task, raw_image, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            result = decoded.upper()
            logger.debug(f"Generated output: {result}")
            return result

    def run_raw_image(self, task: str = None, image_path: str = None, *args, **kwargs):
        logger.debug(f"Running model with task: {task} and image path: {image_path}")
        raw_image = Image.open(image_path)
        inputs = self.processor(task, raw_image, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            result = decoded.upper()
            logger.debug(f"Generated output: {result}")
            return result
