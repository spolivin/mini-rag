from mini_rag.configurations import TextGenerationConfig


def test_text_generation_config():
    gen_params = TextGenerationConfig()
    assert gen_params.max_new_tokens == 250
    assert gen_params.min_new_tokens == 30
    assert gen_params.do_sample == True
    assert gen_params.top_p == 0.9
    assert gen_params.temperature == 0.7
    assert gen_params.no_repeat_ngram_size == 3
