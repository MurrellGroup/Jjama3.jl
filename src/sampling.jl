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
    device = identity
) where T
    current_len = length(initial_tokens)
    tokens = vcat(initial_tokens, similar(initial_tokens, max_new_tokens))

    for layer in model.layers
        config!(layer.attention.cache, seq_length = current_len + max_new_tokens)
    end

    input_tokens = device(reshape(initial_tokens, :, 1))  # (seq_len, batch=1)
    logits = model(input_tokens, 0)
    start_pos = current_len

    # Generate new tokens one at a time
    for _ in 1:max_new_tokens
        # If sequence is empty or we want to process just the last token
        if start_pos == 0
            input_tokens = device(reshape([128001], :, 1))  # Use start of text token if empty
        else
            input_tokens = device(reshape([tokens[current_len]], :, 1))  # Just the last token
        end
        # Get logits for next token
        logits = model(input_tokens, start_pos)
        # Sample next token (logits are size vocab × 1 × 1)
        next_token = sampler(logits[:, end, 1])
        current_len += 1
        tokens[current_len] = next_token
        !isnothing(tokenizer_for_printing) && print(decode(tokenizer_for_printing, [next_token]))
        next_token == end_token && break
        start_pos += 1
    end
    # Clear KV caches
    for layer in model.layers
        clear!(layer.attention.cache)
    end
    return tokens[1:current_len]
end

