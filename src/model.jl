#Note about output layer being tied to embedding: https://github.com/meta-llama/llama-models/issues/172

function create_mask(h::AbstractArray{T}; precached_size = 0) where T<:AbstractFloat
    Flux.ChainRulesCore.ignore() do
        dim, seqlen, batch = size(h)
        mask = similar(h, seqlen, seqlen)
        mask .= T(-Inf)
        mask = tril(mask, -1) #This is swapped because we're using the slightly more efficient dim setup
        if precached_size > 0
            pad = similar(h, precached_size, seqlen)
            pad .= T(0.0)
            mask = vcat(pad, mask)
        end
        return mask
    end
end

function (model::Transformer)(tokens::AbstractArray{Int})
    h = model.tok_embeddings(tokens) # Embedding: (dim, seq_len, batch)
    rope = model.rope[model.pos+1:model.pos+size(tokens, 1)]
    if size(h, 2) == 1
        mask = create_mask(h)
    else
        mask = create_mask(h; precached_size = model.pos)
    end
    for layer in model.layers
        h = layer(h, model.pos, rope, mask)
    end
    h = model.norm(h)
    output = model.output(h)
    model.pos += size(tokens, 1)
    return output
end

function masked_agg(ce, mask)
    if mask !== nothing
        ce = ce .* mask
    end
    return sum(ce)/sum(mask)
end

function forward_loss(model::Transformer, inputs::AbstractArray, 
                     targets::AbstractArray; clear_cache = true, loss_mask = nothing)
    if clear_cache
        Flux.ChainRulesCore.ignore() do
            clear_cache!(model)
        end
    end
    logits = model(inputs)
    vocab_size = size(model.tok_embeddings.weight, 2)
    gt = Flux.onehotbatch(targets, 1:vocab_size)
    if loss_mask !== nothing
        loss = Flux.logitcrossentropy(logits, gt, agg = x -> masked_agg(x, loss_mask))
    else
        loss = Flux.logitcrossentropy(logits, gt)
    end
    return loss
end

# compat
forward_inference(model, args...) = model(args...)
