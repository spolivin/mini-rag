import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..configurations import TextGenerationConfig


class LLMWrapper:
    """A wrapper for loading and interacting with a Large Language Model (LLM)."""

    def __init__(self, model_name: str):
        """Initializes the LLMWrapper with the specified model.

        Args:
            model_name (str): The name of the model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        self.gen_params = TextGenerationConfig().__dict__
        self.system_prompt = (
            "You are a concise assistant. "
            "Use the provided context to answer in 4-5 sentences. "
            "Do not include introductions or sections. If not in the context, say 'I don't know.'"
        )
        self.max_length = getattr(self.model.config, "max_position_embeddings", 4096)

    def _build_prompt(
        self, system_prompt: str, context: list[str], user_query: str
    ) -> str:
        """Builds the prompt for the LLM.

        Args:
            system_prompt (str): System prompt for the LLM.
            context (list[str]): Chunks of context to include.
            user_query (str): User's query.

        Returns:
            str: The formatted prompt for the LLM.
        """
        context_text = "\n\n".join(context)

        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{user_query}",
            },
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            # fallback for base models
            return f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion:\n{user_query}\nAnswer:"

    def _build_llm_inputs(
        self,
        context: list[str],
        user_query: str,
    ) -> dict[str, torch.Tensor]:
        """Builds the inputs for the LLM.

        Args:
            context (list[str]): Chunks of context to include.
            user_query (str): User's query.

        Returns:
            dict[str, torch.Tensor]: Inputs for the LLM.
        """
        prompt = self._build_prompt(
            system_prompt=self.system_prompt, context=context, user_query=user_query
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )

        return input_ids

    def __call__(self, user_query: str, context: list[str]) -> str:
        """Generates a response from the LLM.

        Args:
            user_query (str): The user's query.
            context (list[str]): Chunks of context to include.

        Returns:
            str: The generated response from the LLM.
        """
        print("Generating a response from LLM...")

        input_ids = self._build_llm_inputs(context, user_query)
        model_inputs = input_ids.input_ids.to(self.model.device)
        attention_mask = input_ids.attention_mask.to(self.model.device)
        outputs = self.model.generate(
            model_inputs,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=False,
            **self.gen_params,
        )

        llm_response = self.tokenizer.decode(
            outputs.sequences[0][model_inputs.shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        llm_response = re.sub(r"([a-z])([A-Z])", r"\1 \2", llm_response).strip()

        return llm_response
