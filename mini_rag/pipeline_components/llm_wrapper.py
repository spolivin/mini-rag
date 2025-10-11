import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SUPPORTED_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


class LLMWrapper:
    """A wrapper for loading and interacting with a Large Language Model (LLM).

    This class handles the initialization of the model and tokenizer, as well as the
    construction of prompts and input tensors for generating responses. The loaded
    model is quantized to 4-bit precision using bitsandbytes for efficiency. It also
    includes a system prompt to guide the model's responses. The model is designed to
    provide concise answers based on the provided context.

    Args:
        model_name (str): The name of the model to load.
        textgen_params (dict[str, int | float]): Parameters for text generation.
    """

    SYSTEM_PROMPT = (
        "You are a concise assistant. "
        "Use the provided context to answer in 4-5 sentences. "
        "Do not include introductions or sections. If not in the context, say 'I don't know.'"
    )

    def __init__(self, model_name: str, textgen_params: dict[str, int | float]):
        """Initializes the LLMWrapper with the specified model."""

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not currently supported and using it might lead to unexpected behavior. "
                f"Supported and tested models are: {', '.join(SUPPORTED_MODELS)}"
            )
        # Setting up the tokenizer for encoding and decoding text
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensuring the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configuring the model to use 4-bit quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        self.model.eval()

        self.textgen_params = textgen_params

        # Maximum length for input sequences, defaulting to 4096 if not specified
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
        # Joining context chunks with double newlines for clarity
        context_text = "\n\n".join(context[::-1])

        # # Constructing the conversation structure
        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{user_query}",
            },
        ]

        # Using the tokenizer's chat template
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

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
        # Building the prompt for the LLM
        prompt = self._build_prompt(
            system_prompt=self.SYSTEM_PROMPT, context=context, user_query=user_query
        )

        # Tokenizing the prompt to create input tensors
        llm_input_dict = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )

        return llm_input_dict

    def generate(self, user_query: str, context: list[str]) -> str:
        """Generates a response from the LLM.

        Args:
            user_query (str): The user's query.
            context (list[str]): Chunks of context to include.

        Returns:
            str: The generated response from the LLM.
        """
        print("Generating a response from LLM...")

        # Building inputs for the LLM
        llm_input_dict = self._build_llm_inputs(context=context, user_query=user_query)
        input_ids = llm_input_dict.input_ids.to(self.model.device)
        attention_mask = llm_input_dict.attention_mask.to(self.model.device)

        # Generating the response using the model
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=False,
            **self.textgen_params,
        )

        # Decoding the generated tokens to get the response text
        llm_response = self.tokenizer.decode(
            outputs.sequences[0][input_ids.shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Inserting spaces in camelCase words for better readability (in case model merges words)
        llm_response = re.sub(r"([a-z])([A-Z])", r"\1 \2", llm_response).strip()

        return llm_response
