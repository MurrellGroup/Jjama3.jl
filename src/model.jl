#Note about output layer being tied to embedding: https://github.com/meta-llama/llama-models/issues/172

function masked_agg(ce, mask)
    if mask !== nothing
        ce = ce .* mask
    end
    return sum(ce)/sum(mask)
end

function (model::Transformer)(tokens::AbstractArray{Int}; caches=no_kv_cache(model), kws...)
    h = model.embeddings(tokens)
    rope = model.rope[position(caches) .+ (1:size(tokens, 1))]
    for (layer, cache) in zip(model.layers, caches)
        h = layer(h; rope, cache, kws...)
    end
    h = model.norm(h)
    output = model.output(h)
    return output
end

function loss(logits, targets::AbstractArray; loss_mask = nothing)
    vocab_size = size(logits,1)
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
forward_loss(model::Transformer, inputs::AbstractArray, targets::AbstractArray; clear_cache = true, loss_mask = nothing) = loss(model(inputs, clear_cache = clear_cache), targets, loss_mask = loss_mask)
