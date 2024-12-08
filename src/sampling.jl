# This generate function seems to do one unnecessary forward pass when switching from the forward pass over the initial sequence
# to the sampling of each token. But when I try and fix it, the model gets slightly dumber.
# Vibes feel like a shift-by-1 in the RoPE, or something similar. Need to investigate when I find time.
"""
    generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)

Takes an initial sequence of tokens, and generates new tokens one at a time until the end token is sampled. Uses a KV cache. No batch dim for now.
Runs on CPU by default. If the model is on the GPU (assuming Flux.jl, eg. `model = gpu(model)`), then pass `device = gpu` to `generate` to run on the GPU.

```julia
tkn = llama3_tokenizer()
generate(model, initial_tokens; max_new_tokens=100, sampler=top_pk_sampler(p=0.5f0, k=5), tokenizer_for_printing=tkn, end_token=128010)
```
"""
function generate(
    model::Transformer{T}, 
    initial_tokens::AbstractArray{<:Integer};
    max_new_tokens=100,
    sampler::Function=argmax_sampler,
    tokenizer_for_printing = nothing,
    end_token = 128010,
    clear_cache = true,
    pos_offset = 0,
    device = identity
) where T
    current_len = length(initial_tokens)
    tokens = vcat(initial_tokens, similar(initial_tokens, max_new_tokens))
    if clear_cache
        clear_cache!(model)
        config_cache!(model, current_len + max_new_tokens)
    else
        extend_cache!(model, current_len + max_new_tokens)
    end
    input_tokens = device(reshape(initial_tokens, :, 1))  # (seq_len, batch=1)
    logits = model(input_tokens)
    for _ in 1:max_new_tokens
        input_tokens = device(reshape([tokens[current_len]], :, 1))  # Just the last token
        logits = model(input_tokens)
        next_token = sampler(logits[:, end, 1])
        current_len += 1
        tokens[current_len] = next_token
        !isnothing(tokenizer_for_printing) && print(decode(tokenizer_for_printing, [next_token], skip_special_tokens = false))
        next_token == end_token && break
    end
    return tokens[1:current_len]
end