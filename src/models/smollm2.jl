function smollm2_instruct_prompt(tokenizer, system_prompt, user_prompt)
    str = """<|im_start|>system\n$(system_prompt)<|im_end|>\n<|im_start|>user\n$(user_prompt)<|im_end|>\n<|im_start|>assistant\n"""
    return encode(tokenizer, str)
end

smollm2_assistant_prompt(tokenizer, prompt) = smollm2_instruct_prompt(tokenizer, "You are a helpful AI assistant named SmolLM, trained by Hugging Face", prompt);
