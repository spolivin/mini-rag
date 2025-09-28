import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..configurations import TextGenerationConfig


class AnswerGenerator:
    """Answer text LLM-based generator.

    Args:
        model_name (str): Name of the text generation model.
        gen_params (TextGenerationConfig): Config for text generation parameters.
        task (str, optional): Task being accomplished by a model. Defaults to "text2text-generation".
    """

    def __init__(
        self,
        model_name: str,
        gen_params: TextGenerationConfig,
    ):
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
        self.gen_params = gen_params.__dict__
        self.system_prompt = (
            "[INST] <<SYS>> You are a concise assistant. "
            "Use the provided context to answer in 1-2 sentences. "
            "Do not include introductions or sections. If not in the context, say 'I don't know.'\n"
            "<</SYS>>\n\n"
        )

    def _build_llama_inputs(
        self,
        context: str,
        user_query: str,
        max_length: int = None,
    ):
        if max_length is None:
            if self.model is not None and hasattr(
                self.model.config, "max_position_embeddings"
            ):
                max_length = self.model.config.max_position_embeddings
            else:
                max_length = 4096

        sys_ids = self.tokenizer(self.system_prompt, add_special_tokens=False)[
            "input_ids"
        ]
        query_ids = self.tokenizer(user_query, add_special_tokens=False)["input_ids"]

        budget = max_length - (len(sys_ids) + len(query_ids))
        if budget <= 0:
            raise ValueError("System prompt + query exceed max length!")

        context = "\n\n".join(context[::-1])
        context = "Context:\n" + context
        ctx_ids = self.tokenizer(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=budget,
        )["input_ids"]

        final_ids = sys_ids + ctx_ids + query_ids

        return {"input_ids": final_ids}

    def __call__(self, query: str, context: list[str]) -> str:
        """Generates an answer using the LLM based on the provided prompt.

        Args:
            prompt (str): The prompt containing context and the user's question.

        Returns:
            str: Generated answer.
        """
        print("Generating a response from LLM...")
        query = f"Question:\n{query}\n[/INST]"

        inputs = self._build_llama_inputs(context, query)
        input_ids = torch.tensor([inputs["input_ids"]]).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            return_dict_in_generate=True,
            output_scores=False,
            **self.gen_params,
        )

        return self.tokenizer.decode(
            outputs.sequences[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
