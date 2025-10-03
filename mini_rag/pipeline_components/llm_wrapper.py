import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..configurations import TextGenerationConfig


class LLMWrapper:

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        self.gen_params = TextGenerationConfig().__dict__
        self.system_prompt = (
            "You are a concise assistant. "
            "Use the provided context to answer in 4-5 sentences. "
            "Do not include introductions or sections. If not in the context, say 'I don't know.'"
        )
        self.max_length = getattr(self.model.config, "max_position_embeddings", 4096)

    def _build_prompt(self, system, context_chunks, query):
        context_text = "\n\n".join(context_chunks)

        conversation = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{query}",
            },
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        else:
            # fallback for base models
            return (
                f"{system}\n\nContext:\n{context_text}\n\nQuestion:\n{query}\nAnswer:"
            )

    def _build_llama_inputs(
        self,
        context: str,
        user_query: str,
    ):
        prompt = self._build_prompt(self.system_prompt, context, user_query)
        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.to(self.model.device)

        return input_ids

    def __call__(self, query: str, context: list[str]) -> str:
        print("Generating a response from LLM...")

        input_ids = self._build_llama_inputs(context, query)

        outputs = self.model.generate(
            input_ids,
            return_dict_in_generate=True,
            output_scores=False,
            **self.gen_params,
        )

        return self.tokenizer.decode(
            outputs.sequences[0][input_ids.shape[-1] :], skip_special_tokens=True
        )
