function nexttoken!(tokens, pos, model, sampler, logits, tokenizer_for_printing)
    tokens[pos+1:pos+1] .= argmax_sampler(logits[:, end, 1])
    !isnothing(tokenizer_for_printing) && print(decode(tokenizer_for_printing, tokens[pos+1:pos+1] |> cpu, skip_special_tokens = false))
end

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
    io=stdout,
    max_new_tokens=100,
    sampler::Function=argmax_sampler, #top_pk_sampler(p=0.5f0, k=5),
    tokenizer_for_printing = nothing,
    end_token = 128010,
    caches=kv_cache(model, 1024, 1),
    kws...
) where T
    n, b = size(initial_tokens, 1), size(initial_tokens, 2)
    tokens = reshape(initial_tokens, n, b)
    n > 1 && model(tokens[1:n-1, :]; caches, mask=causal_mask, kws...)
    for i in 1:max_new_tokens
        logits = model(tokens[end:end, 1]; caches, kws...)
        tokens = [tokens; sampler(logits[:, end])]
        !isnothing(tokenizer_for_printing) && print(io, decode(tokenizer_for_printing, tokens[end:end] |> cpu, skip_special_tokens = false))
        sum(tokens[end:end]) == end_token && break
    end
    return tokens
end
