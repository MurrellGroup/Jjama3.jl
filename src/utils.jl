using Accessors

encode(tkn::Tokenizer, str; kwargs...) = HuggingFaceTokenizers.encode(tkn, str; kwargs...).ids .+ 1
decode(tkn::Tokenizer, ids; kwargs...) = HuggingFaceTokenizers.decode(tkn, ids .- 1; kwargs...)

#Trivial Char tokenizers:
encode(chars::Vector{Char}, str::String) = [findfirst(==(c), chars) for c in str]
decode(chars::Vector{Char}, enc::Vector{Int}; skip_special_tokens=false) = String([chars[i] for i in enc])

#For training:
function pad_and_batch(seqs, pad_token)
    max_len = maximum(length.(seqs))
    padded = [vcat(s, fill(pad_token, max_len - length(s))) for s in seqs]
    cat(padded..., dims = 2)
end


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
    str = """<|im_start|>system\n$(system_prompt)<|im_end|>\n<|im_start|>user\n$(user_prompt)<|im_end|>\n<|im_start|>assistant\n"""
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
function load_llama3_from_safetensors(
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
    if config[:model_type] == "qwen2"
        qkv_bias = true
    else
        qkv_bias = false
    end
    model = Transformer(
        config[:vocab_size],                        # vocab_size
        config[:hidden_size],                       # dim (hidden_size)
        config[:num_hidden_layers],                 # n_layers (num_hidden_layers)
        config[:num_attention_heads],               # n_heads (num_attention_heads)
        config[:num_key_value_heads],               # n_kv_heads (num_key_value_heads)
        config[:max_position_embeddings],           # max_seq_len (max_position_embeddings)
        config[:intermediate_size],                 # ff_hidden_dim
        qkv_bias = qkv_bias,                        # qkv_bias
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

            if haskey(weights, "$prefix.self_attn.q_proj.bias")
                layer.attention.wq.bias .= weights["$prefix.self_attn.q_proj.bias"]
            end
            if haskey(weights, "$prefix.self_attn.k_proj.bias")
                layer.attention.wk.bias .= weights["$prefix.self_attn.k_proj.bias"]
            end
            if haskey(weights, "$prefix.self_attn.v_proj.bias")
                layer.attention.wv.bias .= weights["$prefix.self_attn.v_proj.bias"]
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

    if !isempty(add_lora_to)
        if :Q in add_lora_to
            for layer in model.layers
                @reset layer.attention.wq = LoRADense(layer.attention.wq, lora_dim)
            end
        end
        if :K in add_lora_to
            for layer in model.layers
                @reset layer.attention.wk = LoRADense(layer.attention.wk, lora_dim)
            end
        end
        if :V in add_lora_to
            for layer in model.layers
                @reset layer.attention.wv = LoRADense(layer.attention.wv, lora_dim)
            end
        end
        if :O in add_lora_to
            for layer in model.layers
                @reset layer.attention.wo = LoRADense(layer.attention.wo, lora_dim)
            end
        end
        if :w1 in add_lora_to
            for layer in model.layers
                @reset layer.feed_forward.w1 = LoRADense(layer.feed_forward.w1, lora_dim)
            end
        end
        if :w2 in add_lora_to
            for layer in model.layers
                @reset layer.feed_forward.w2 = LoRADense(layer.feed_forward.w2, lora_dim)
            end
        end
        if :w3 in add_lora_to
            for layer in model.layers
                @reset layer.feed_forward.w3 = LoRADense(layer.feed_forward.w3, lora_dim)
            end
        end
    end
    return model
end

load_llama3_from_safetensors(path::String, config; T = Float32, kwargs...) = load_llama3_from_safetensors([path], config; T = T, kwargs...)


"""
    sampler = structured_choice(choices, vocab::Vector{String}, end_token::Int; sampler = logits -> argmax_sampler(logits))

Return a function that can be passed into generate as a sampler, which will sample from the given choices. Handles the case where the choices are made up of multiple tokens.
`vocab` is an array of the tokens as strings, in their order in the tokenizer. `sampler` is a function that takes the logits (here including those masked with -Inf) and returns a sample from them. Defaults to argmax.

Example:
```julia
config = JSON3.read(read("SmolLM2-1.7B-Instruct/config.json", String))
model = load_llama3_from_safetensors("SmolLM2-1.7B-Instruct/model.safetensors", config)
tkn = tokenizer_from_file(Tokenizer, "SmolLM2-1.7B-Instruct/tokenizer.json")

question = "In a Bayesian model, what do we call the probability distribution of parameters given the data?"
choices = ["Prior", "Likelihood", "Marginal Likelihood", "Evidence", "Posterior"]

vocab = [decode(tkn, [i], skip_special_tokens = false) for i in 1:49152]
eos = encode(tkn, "<|im_end|>")[end]
prompt = smollm2_instruct_prompt(tkn, "You are an expert in Statistics and Probability Theory who answers questions in as few words as possible.",question)
generate(model, prompt, max_new_tokens=100, tokenizer_for_printing=tkn, end_token = eos, sampler = structured_choice(choices, vocab, eos));
```

If you want to run the model on the GPU, then you need to pass `device = gpu` to the `generate` function, and `device = cpu` to the `structured_choice` function.
"""
function structured_choice(choices::Vector{String}, vocab::Vector{String}, end_token::Int; sampler = logits -> argmax_sampler(logits), device = identity)
    remaining_choices = copy(choices)
    function choice_sampler(logits)
        logits = device(logits)
        if length(remaining_choices) == 0 || maximum(length.(remaining_choices)) == 0
            return end_token
        end
        mask = zeros(Bool, length(vocab))
        for i in 1:length(vocab)
            for choice in remaining_choices
                if startswith(choice, vocab[i])
                    mask[i] = true
                end
            end
        end
        logits[.!mask] .= -Inf
        next_token = sampler(logits)
        next_token_str = vocab[next_token]
        remaining_choices = [choice[length(next_token_str)+1:end] for choice in remaining_choices if startswith(choice, next_token_str)]
        return next_token
    end
    return choice_sampler
end


function export_model(model, output_path; type_convert = identity)
    weights = Dict{String,AbstractArray}()
    weights["model.embed_tokens.weight"] = type_convert(model.tok_embeddings.weight')
    weights["lm_head.weight"] = type_convert(model.output.weight)
    weights["model.norm.weight"] = type_convert(model.norm.weight)
    for (i, layer) in enumerate(model.layers)
        prefix = "model.layers.$(i-1)"
        weights["$prefix.self_attn.q_proj.weight"] = type_convert(layer.attention.wq.weight)
        weights["$prefix.self_attn.k_proj.weight"] = type_convert(layer.attention.wk.weight)
        weights["$prefix.self_attn.v_proj.weight"] = type_convert(layer.attention.wv.weight)
        weights["$prefix.self_attn.o_proj.weight"] = type_convert(layer.attention.wo.weight)
        if layer.attention.wq.bias
            weights["$prefix.self_attn.q_proj.bias"] = type_convert(layer.attention.wq.bias)
            weights["$prefix.self_attn.k_proj.bias"] = type_convert(layer.attention.wk.bias)
            weights["$prefix.self_attn.v_proj.bias"] = type_convert(layer.attention.wv.bias)
        end
        weights["$prefix.mlp.gate_proj.weight"] = type_convert(layer.feed_forward.w1.weight)
        weights["$prefix.mlp.down_proj.weight"] = type_convert(layer.feed_forward.w2.weight)
        weights["$prefix.mlp.up_proj.weight"] = type_convert(layer.feed_forward.w3.weight)
        weights["$prefix.input_layernorm.weight"] = type_convert(layer.attention_norm.weight)
        weights["$prefix.post_attention_layernorm.weight"] = type_convert(layer.ffn_norm.weight)
    end
    SafeTensors.serialize(output_path, weights)
    return nothing
end

#=
#Example for how to save a model in safetensors format
julia> using Random, BFloat16s, SafeTensors
julia> weights = Dict("W"=>randn(BFloat16, 3, 5), "b"=>rand(BFloat16, 3))
Dict{String, Array{BFloat16}} with 2 entries:
  "W" => [0.617188 0.695312 … 0.390625 -2.0; -0.65625 -0.617188 … 0.652344 0.244141; 0.226562 2.70312 … -0.174805 -0.7773…
  "b" => [0.111816, 0.566406, 0.283203]
julia> f = tempname();
julia> SafeTensors.serialize(f, weights)
=#