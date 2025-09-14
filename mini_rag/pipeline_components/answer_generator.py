from transformers import pipeline

from ..configurations import TextGenerationConfig
from ..preprocessing import prettify_answer


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
        task: str = "text2text-generation",
    ):
        self.model = pipeline(task=task, model=model_name)
        self.gen_params = gen_params.__dict__

    def __call__(self, prompt: str) -> str:
        """Generates an answer using the LLM based on the provided prompt.

        Args:
            prompt (str): The prompt containing context and the user's question.

        Returns:
            str: Generated answer.
        """
        print("Generating a response from LLM...")
        answer = self.model(prompt, **self.gen_params)
        answer = answer[0]["generated_text"]

        return prettify_answer(answer)
