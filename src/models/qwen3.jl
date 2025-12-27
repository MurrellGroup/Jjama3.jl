
function qwen3_instruct_prompt(tokenizer,system_prompt, user_prompt)
    str = "<|im_start|>system\n$(system_prompt)<|im_end|>\n<|im_start|>user\n$(user_prompt)<|im_end|>\n<|im_start|>assistant\n"
    return encode(tokenizer, str)
end

qwen3_assistant_prompt(tokenizer, prompt) = qwen3_instruct_prompt(tokenizer,"\nYou are a helpful assistant\n", prompt);

function load_qwen3_from_safetensors(
    paths::Vector{String}, config;
    T = Float32, add_lora_to = Symbol[], lora_dim = 0,
)
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
    qkv_bias = config[:model_type] == "qwen2"
    qk_norm = config[:model_type] == "qwen3"
    head_dim_kwarg = haskey(config, :head_dim) ? (; head_dim = config[:head_dim]) : (;)
    model = Transformer(
        config[:vocab_size],                        # vocab_size
        config[:hidden_size],                       # dim (hidden_size)
        config[:num_hidden_layers],                 # n_layers (num_hidden_layers)
        config[:num_attention_heads],               # n_heads (num_attention_heads)
        config[:num_key_value_heads],               # n_kv_heads (num_key_value_heads)
        config[:max_position_embeddings],           # max_seq_len (max_position_embeddings)
        config[:intermediate_size];                 # ff_hidden_dim
        head_dim_kwarg...,
        qk_norm,
        qkv_bias,
        norm_eps=T(config[:rms_norm_eps]),          # rms_norm_eps
        rope_settings = (;
            theta = T(config[:rope_theta]),
            use_scaled = true,
            scale_factor),
    )
    
    for path in paths # Process one file at a time
        weights = load_safetensors(path)
        #if (haskey(weights, "lm_head.weight") && (config[:tie_word_embeddings]))
        #    error("tie_word_embeddings was true, but lm_head.weight was present.")
        #end
        if haskey(weights, "model.embed_tokens.weight")
            model.embeddings.weight .= weights["model.embed_tokens.weight"]'
            if config[:tie_word_embeddings]
                model.output.weight .= weights["model.embed_tokens.weight"]
            else
                model.output.weight .= weights["lm_head.weight"]
            end
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

            if haskey(weights, "$prefix.self_attn.q_proj.bias")
                layer.attention.wq.bias .= weights["$prefix.self_attn.q_proj.bias"]
            end
            if haskey(weights, "$prefix.self_attn.k_proj.bias")
                layer.attention.wk.bias .= weights["$prefix.self_attn.k_proj.bias"]
            end
            if haskey(weights, "$prefix.self_attn.v_proj.bias")
                layer.attention.wv.bias .= weights["$prefix.self_attn.v_proj.bias"]
            end

            if haskey(weights, "$prefix.self_attn.q_norm.weight")
                layer.attention.q_norm.weight .= weights["$prefix.self_attn.q_norm.weight"]
            end
            if haskey(weights, "$prefix.self_attn.k_norm.weight")
                layer.attention.k_norm.weight .= weights["$prefix.self_attn.k_norm.weight"]
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

load_qwen3_from_safetensors(path::String, config; T = Float32, kws...) = load_qwen3_from_safetensors([path], config; T = T, kws...)
