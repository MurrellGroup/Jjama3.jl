#Note about output layer being tied to embedding: https://github.com/meta-llama/llama-models/issues/172

function create_mask(h::AbstractArray{T}) where T<:AbstractFloat
    dim, seqlen, batch = size(h)
    mask = similar(h, seqlen, seqlen)
    mask .= T(-Inf)
    #mask = triu(mask, 1)
    mask = tril(mask, -1) #This is swapped because we're using the slightly more efficient dim setup
    return mask
end

function (model::Transformer)(tokens::AbstractArray{Int}, start_pos::Int=0)
    h = model.tok_embeddings(tokens) # Embedding: (dim, seq_len, batch)
    rope = model.rope[start_pos+1:start_pos+size(tokens, 1)]
    mask = create_mask(h)
    for layer in model.layers
        h = layer(h, start_pos, rope, mask)
    end
    h = model.norm(h)
    output = model.output(h)
    return output
end

function forward_loss(model::Transformer, inputs::AbstractArray, 
                     targets::AbstractArray; ignore_index::Int=-100,
                     mask = :auto)
    seqlen = size(inputs, 1) #(seq_len, batch)
    h = model.tok_embeddings(inputs) # (dim, seq_len, batch)
    rope = model.rope[1:seqlen]
    mask = create_mask(h)
    for layer in model.layers
        h = layer(h, 0, rope, mask)
    end
    h = model.norm(h)
    logits = model.output(h)
    # Need to reshape to (vocab_size, seq_len * batch)
    logits_2d = reshape(logits, size(logits,1), :)
    targets_1d = reshape(targets, :)
    # Mask out ignored indices - will handle this later.
    # Note: this is not the autoregressive mask, but the mask for the loss function
    #=
    mask = targets_1d .!= ignore_index
    if any(mask)
        loss = Flux.logitcrossentropy(
            logits_2d[:, mask],
            targets_1d[mask]
        )
    else
        loss = zero(Float32)
    end
    =#
    vocab_size = size(model.tok_embeddings.weight, 2)
    gt = Flux.onehotbatch(targets_1d, 1:vocab_size)
    loss = Flux.logitcrossentropy(logits_2d, gt)
    return loss
end

# compat
forward_inference(model, args...) = model(args...)
