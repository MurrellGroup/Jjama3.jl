var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Jjama3","category":"page"},{"location":"#Jjama3","page":"Home","title":"Jjama3","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Jjama3.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Jjama3]","category":"page"},{"location":"#Jjama3.assistant_prompt-Tuple{Any, Any}","page":"Home","title":"Jjama3.assistant_prompt","text":"Format a prompt for use with Llama3.2's instruction format, with a simple \"You are a helpful assistant\" system prompt.\n\ntkn = llama3_tokenizer()\nprompt = assistant_prompt(\"What is the capital of France?\", tkn)\ngenerate(model, prompt, max_new_tokens=100, encoder_for_printing=tkn)\n\n\n\n\n\n","category":"method"},{"location":"#Jjama3.format_llama32_instruction_prompt-Tuple{Any, Any, Any}","page":"Home","title":"Jjama3.format_llama32_instruction_prompt","text":"Format a prompt for use with Llama3.2's instruction format, injecting the system and user roles.\n\ntkn = llama3_tokenizer()\nprompt = format_llama32_instruction_prompt(\"\\nYou are a helpful assistant\\n\", \"What is the capital of France?\", tkn)\ngenerate(model, prompt, max_new_tokens=100, encoder_for_printing=tkn)\n\n\n\n\n\n","category":"method"},{"location":"#Jjama3.generate-Union{Tuple{IntT}, Tuple{T}, Tuple{Jjama3.Transformer{T}, AbstractArray{IntT}}} where {T, IntT}","page":"Home","title":"Jjama3.generate","text":"Takes an initial sequence of tokens, and generates new tokens one at a time until the end token is sampled. Uses a KV cache. No batch dim for now. Runs on CPU by default. If the model is on the GPU (assuming Flux.jl, eg. model = gpu(model)), then pass device = gpu to generate to run on the GPU.\n\ntkn = llama3_tokenizer()\ngenerate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), encoder_for_printing=tkn, end_token=128010)\n\n\n\n\n\n","category":"method"},{"location":"#Jjama3.load_llama3_from_safetensors-Tuple{Vector{String}, Any}","page":"Home","title":"Jjama3.load_llama3_from_safetensors","text":"Load a Llama3 model from a set of Huggingface safetensors files, and the config.json file. Important note: Huggingface uses a different RoPE convention than other implementations, so if you're loading weights from a different source, you might get very poor model performance.\n\nusing JSON3\nconfig = JSON3.read(read(\"Llama3_2_1B_instruct/config.json\", String))\nmodel_weight_paths = [\"Llama3_2_1B_instruct/model.safetensors\"] #Can be an array of paths if the model is split across multiple files\nmodel = load_llama3_from_safetensors(model_weight_paths, config)\n\n\n\n\n\n","category":"method"}]
}
