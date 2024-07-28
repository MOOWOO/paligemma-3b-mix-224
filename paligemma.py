import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from transformers import BitsAndBytesConfig
from swarms import BaseMultiModalModel
from huggingface_hub import login
import os
from dotenv import load_dotenv

torch.set_default_device("cuda")

load_dotenv()
access_token = os.getenv('HF_TOKEN')
login(access_token)

class PaliGemma(BaseMultiModalModel):
    """
    PaliGemma is a class that represents a model for conditional generation using the PaliGemma model.

    Args:
        model_id (str): The identifier of the PaliGemma model to be used. Default is "google/paligemma-3b-mix-224".
        max_new_tokens (int): The maximum number of new tokens to be generated. Default is 20.
        skip_special_tokens (bool): Whether to skip special tokens during decoding. Default is True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        model_id (str): The identifier of the PaliGemma model.
        model (PaliGemmaForConditionalGeneration): The PaliGemma model for conditional generation.
        processor (AutoProcessor): The processor for the PaliGemma model.
        max_new_tokens (int): The maximum number of new tokens to be generated.
        skip_special_tokens (bool): Whether to skip special tokens during decoding.

    Methods:
        run: Runs the PaliGemma model for conditional generation.

    """

    def __init__(
        self,
        model_id: str = "google/paligemma-3b-mix-224",
        max_new_tokens: int = 50,
        skip_special_tokens: bool = True,
        *args,
        **kwargs
    ):
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

    def run(self, task: str = None, image_url: str = None, *args, **kwargs):
        """
        Runs the PaliGemma model for conditional generation.

        Args:
            task (str): The task or prompt for conditional generation.
            image_url (str): The URL of the image to be used as input.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The generated output text.

        """
        raw_image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.processor(task, raw_image, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens , do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            return decoded.upper()

    def run_raw_image(self, task: str = None, image_path: str = None, *args, **kwargs):
        """
        Runs the PaliGemma model for conditional generation.

        Args:
            task (str): The task or prompt for conditional generation.
            image_path (str): The path of the image to be used as input.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The generated output text.

        """
        raw_image = Image.open(image_path)
        inputs = self.processor(task, raw_image, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            return decoded.upper()