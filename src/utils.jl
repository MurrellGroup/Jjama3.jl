encode(tkn::Tokenizer, str) = HuggingFaceTokenizers.encode(tkn, str).ids .+ 1
decode(tkn::Tokenizer, ids) = HuggingFaceTokenizers.decode(tkn, ids .- 1)


#https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
function llama3_instruct_prompt(tokenizer,system_prompt, user_prompt)
    str = """<|start_header_id|>system<|end_header_id|>
$system_prompt
<|eot_id|><|start_header_id|>user<|end_header_id|>

$(user_prompt)<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return encode(tokenizer, str)
end

"""
    generate(model, prompt, max_new_tokens=100, encoder_for_printing=tkn)

Format a prompt for use with Llama3.2's instruction format, with a simple "You are a helpful assistant" system prompt.

    prompt = assistant_prompt(tkn, "What is the capital of France?")
    generate(model, prompt, max_new_tokens=100, encoder_for_printing=tkn)
"""
llama3_assistant_prompt(tokenizer, prompt) = llama3_instruct_prompt(tokenizer,"\nYou are a helpful assistant\n", prompt);

function smollm2_instruct_prompt(tokenizer, system_prompt, user_prompt)
    str = """<|im_start|>system\n$(system_prompt)<|im_end|>\n<|im_start|>user\n$(user_prompt)<|im_end|>\n"""
    return encode(tokenizer, str)
end

smollm2_assistant_prompt(tokenizer, prompt) = smollm2_instruct_prompt(tokenizer, "You are a helpful AI assistant named SmolLM, trained by Hugging Face", prompt);



"""
    model = load_llama3_from_safetensors(model_weight_paths, config)

Load a Llama3 model from a set of Huggingface safetensors files, and the config.json file.
Important note: Huggingface uses a different RoPE convention than other implementations,
so if you're loading weights from a different source, you might get very poor model performance.

    using JSON3
    config = JSON3.read(read("Llama3_2_1B_instruct/config.json", String))
    model_weight_paths = ["Llama3_2_1B_instruct/model.safetensors"] #Can be an array of paths if the model is split across multiple files
    model = load_llama3_from_safetensors(model_weight_paths, config)
"""
function load_llama3_from_safetensors(paths::Vector{String}, config; T = Float32)
    config = Dict(config) #Just in case the user passed eg. a JSON3.Object
    #@assert config[:rope_scaling][:rope_type] == "llama3"
    #@assert config[:rope_scaling][:low_freq_factor] == 1
    #@assert config[:rope_scaling][:high_freq_factor] == 4
    #@assert config[:rope_scaling][:original_max_position_embeddings] == 8192

    # Create model with config parameters from the JSON
    scale_factor = 1f0
    if haskey(config, :rope_scaling)
        if !isnothing(config[:rope_scaling])
            scale_factor = config[:rope_scaling][:factor]
        end
    end
    model = Transformer(
        config[:vocab_size],                        # vocab_size
        config[:hidden_size],                       # dim (hidden_size)
        config[:num_hidden_layers],                 # n_layers (num_hidden_layers)
        config[:num_attention_heads],               # n_heads (num_attention_heads)
        config[:num_key_value_heads],               # n_kv_heads (num_key_value_heads)
        config[:max_position_embeddings],           # max_seq_len (max_position_embeddings)
        config[:intermediate_size],                 # ff_hidden_dim
        norm_eps=T(config[:rms_norm_eps]),          # rms_norm_eps
        rope_theta=T(config[:rope_theta]),          # rope_theta
        use_scaled_rope=true,                       # Using scaled RoPE based on the config
        scale_factor=scale_factor                   # scale_factor
    )
    
    for path in paths # Process one file at a time
        weights = load_safetensors(path)
        if (haskey(weights, "lm_head.weight") && (config[:tie_word_embeddings]))
            error("tie_word_embeddings was true, but lm_head.weight was present.")
        end
        if haskey(weights, "model.embed_tokens.weight")
            model.tok_embeddings.weight .= weights["model.embed_tokens.weight"]'
            if config[:tie_word_embeddings]
                model.output.weight .= weights["model.embed_tokens.weight"]
            end
        end
        if haskey(weights, "lm_head.weight")
            model.output.weight .= weights["lm_head.weight"]
        end
        if haskey(weights, "model.norm.weight")
            model.norm.weight .= weights["model.norm.weight"]
        end
        
        n_layers = length(model.layers)
        for i in 0:(n_layers-1)
            prefix = "model.layers.$i"
            layer = model.layers[i+1]
  
            if haskey(weights, "$prefix.self_attn.q_proj.weight")
                layer.attention.wq.weight .= weights["$prefix.self_attn.q_proj.weight"]
            end
            if haskey(weights, "$prefix.self_attn.k_proj.weight")
                layer.attention.wk.weight .= weights["$prefix.self_attn.k_proj.weight"]
            end
            if haskey(weights, "$prefix.self_attn.v_proj.weight")
                layer.attention.wv.weight .= weights["$prefix.self_attn.v_proj.weight"]
            end
            if haskey(weights, "$prefix.self_attn.o_proj.weight")
                layer.attention.wo.weight .= weights["$prefix.self_attn.o_proj.weight"]
            end
            
            if haskey(weights, "$prefix.mlp.gate_proj.weight")
                layer.feed_forward.w1.weight .= weights["$prefix.mlp.gate_proj.weight"]
            end
            if haskey(weights, "$prefix.mlp.down_proj.weight")
                layer.feed_forward.w2.weight .= weights["$prefix.mlp.down_proj.weight"]
            end
            if haskey(weights, "$prefix.mlp.up_proj.weight")
                layer.feed_forward.w3.weight .= weights["$prefix.mlp.up_proj.weight"]
            end
            
            if haskey(weights, "$prefix.input_layernorm.weight")
                layer.attention_norm.weight .= weights["$prefix.input_layernorm.weight"]
            end
            if haskey(weights, "$prefix.post_attention_layernorm.weight")
                layer.ffn_norm.weight .= weights["$prefix.post_attention_layernorm.weight"]
            end
        end
        
        weights = nothing
        GC.gc()
    end
    
    return model
end

load_llama3_from_safetensors(path::String, config; T = Float32) = load_llama3_from_safetensors([path], config; T = T)

