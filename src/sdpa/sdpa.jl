include("NNop.jl")

function create_causal_mask(h::AbstractArray{T}; precached_size = 0) where T<:AbstractFloat
    @ignore_derivatives begin
        dim, seqlen = size(h)
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

# Scaled dot product attention
function sdpa(q::AbstractArray{T}, k::AbstractArray{T}, v::AbstractArray{T}; mask=false) where T
    input_size = size(q)
    q, k, v = rearrange.((q, k, v), einops"d l ... -> d l (...)")
    d = size(q, 1)
    mask = mask === causal_mask ? create_causal_mask(q) : mask
    A = softmax(batched_mul(batched_transpose(k), q) / √T(d) .+ mask, dims=1)
    x = batched_mul(v, A)
    return reshape(x, input_size)
end


function keychunked_sdpa(xq::AbstractArray{T,4},
                      xk::AbstractArray{T,4},
                      xv::AbstractArray{T,4};
                      mask=causal_mask,
                      k_chunk_size::Int=128
                     ) where {T<:Real}
    input_size = size(xq)
    head_dim = size(xq, 1)
    xq, xk, xv = rearrange.((xq, xk, xv), einops"d l ... -> d l (...)")
    mask = mask === causal_mask ? create_causal_mask(xq) : mask

    k_len  = size(xk,2)
    q_len  = size(xq,2)
    nbatch = size(xq,3)

    scale = one(T) / √T(head_dim)
    
    partial_max  = fill!(similar(xq, 1, q_len, nbatch), -Inf)
    partial_expw = fill!(similar(xq, 1, q_len, nbatch), T(0))
    partial_vals = fill!(similar(xq, head_dim, q_len, nbatch), T(0))

    # Preallocate local buffers for each chunk
    attn      = fill!(similar(xq, k_chunk_size, q_len, nbatch), T(0))
    local_max = fill!(similar(xq, 1, q_len, nbatch), T(0))
    new_max   = similar(local_max)
    w_old     = similar(local_max)
    w_new     = similar(local_max)
    chunk_sum = similar(local_max)
    valpart   = fill!(similar(xq, head_dim, q_len, nbatch), T(0))

    kstart = 1
    while kstart <= k_len
        k_batch = min(k_chunk_size, k_len - kstart + 1)
        xk_chunk   = @view xk[:, kstart : kstart + k_batch - 1, :]
        xv_chunk   = @view xv[:, kstart : kstart + k_batch - 1, :]
        if length(mask) > 1
            mask_chunk = @view mask[kstart : kstart + k_batch - 1, :, :]
        else
            mask_chunk = mask #Handles the case where the mask is 1-by-1 for sampling a single token.
        end
        attn_view = @view attn[1:k_batch, 1:q_len, 1:nbatch]
        xkT_chunk = batched_transpose(xk_chunk)

        batched_mul!(attn_view, xkT_chunk, xq, scale, 0)  # attn_view = scale*(xkT_chunk*xq)
        attn_view .= attn_view .+ mask_chunk  # add mask

        local_max .= maximum(attn_view, dims=1)
        @. new_max = max(partial_max, local_max)
        @. w_old = exp(partial_max - new_max)
        @. w_new = exp(local_max   - new_max)
        @. attn_view = exp(attn_view - local_max)

        partial_vals .= partial_vals .* w_old # Rescale old accumulators by w_old
        partial_expw .= partial_expw .* w_old

        chunk_sum .= sum(attn_view, dims=1) .* w_new
        partial_expw .+= chunk_sum

        batched_mul!(valpart, xv_chunk, attn_view)
        valpart .= valpart .* w_new
        partial_vals .+= valpart
        partial_max .= new_max
        kstart += k_batch
    end

    y = partial_vals ./ partial_expw
    return reshape(y, input_size)
end

#=
#Todo: use this to ignore parts of the -Inf mask triangle, since we're processing over chunks of queries.
function querychunked_sdpa(
    xq::AbstractArray{T,4},
    xk::AbstractArray{T,4},
    xv::AbstractArray{T,4};
    mask=causal_mask,
    q_chunk_size::Int=128
) where {T<:Real}
    input_size = size(xq)
    head_dim = size(xq, 1)
    xq, xk, xv = rearrange.((xq, xk, xv), einops"d l ... -> d l (...)")
    mask = mask === causal_mask ? create_causal_mask(xq) : mask
    # FIXME: this method is trying to view the mask which fails for `mask::Bool`
    q_len   = size(xq, 2)
    kv_len  = size(xv, 2)
    nbatch  = size(xq, 3)
    q_chunk_size = min(q_chunk_size, q_len)
    α = sqrt(T(head_dim))
    y = similar(xq)
    qk_chunk = similar(xq, kv_len, q_chunk_size, nbatch)
    Achunk = similar(xq, kv_len, q_chunk_size, nbatch)
    qstart = 1
    while qstart <= q_len
        q_batch = min(q_chunk_size, q_len - qstart + 1)
        qinds = qstart:qstart+q_batch-1
        qk_chunkview = view(qk_chunk,:,1:q_batch,:)
        batched_mul!(qk_chunkview,batched_transpose(xk), view(xq, :, qinds, :), 1/α)
        Achunk[:,1:q_batch,:] .= softmax((qk_chunkview .+ view(mask,:,qinds)); dims=1) #(LKV, LQ, HB) "head-batch"
        batched_mul!(view(y,:,qinds,:),xv, view(Achunk,:,1:q_batch,:)) #(D, LQ, HB)
        qstart += q_batch
    end
    return reshape(y, input_size)
end
=#

#=

#Will use Zygote - for testing grad correctness:
function sdpa_norrule(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, mask::AbstractArray{T}, head_dim::Int) where T
    A = softmax(batched_mul(batched_transpose(xk), xq) / sqrt(T(head_dim)) .+ mask; dims=1)
    return batched_mul(xv, A)
end

function ChainRulesCore.rrule(::typeof(sdpa),
                              xq::AbstractArray{T}, #(D, LQ, HB)
                              xk::AbstractArray{T}, #(D, LKV, HB)
                              xv::AbstractArray{T}, #(D, LKV, HB)
                              mask::AbstractArray{T}, #(LKV, LQ)
                              head_dim::Int
                              ) where {T}
    α = sqrt(T(head_dim))
    A = softmax(((batched_mul(batched_transpose(xk), xq) ./ α) .+ mask); dims=1) #(LKV, LQ, HB) "head-batch"
    y = batched_mul(xv, A) #(D, LQ, HB)
    function sdpa_pullback(ȳ)
        xv̄ = batched_mul(ȳ, batched_transpose(A)) #(D, LKV, HB)
        Ā  = batched_mul(batched_transpose(xv), ȳ) #(LKV, LQ, HB)
        dM = (A .* (Ā .- (sum(A .* Ā, dims=1)))) ./ α #(LKV, LQ, HB)
        xq̄ = batched_mul(xk, dM) #(D, LQ, HB)
        xk̄ = batched_mul(xq, batched_transpose(dM)) #(D, LKV, HB)
        return NoTangent(), xq̄, xk̄, xv̄, NoTangent(), NoTangent()
    end
    return y, sdpa_pullback
end


function ChainRulesCore.rrule(::typeof(querychunked_sdpa),
                              xq::AbstractArray{T}, #(D, LQ, HB)
                              xk::AbstractArray{T}, #(D, LKV, HB)
                              xv::AbstractArray{T}, #(D, LKV, HB)
                              mask::AbstractArray{T}, #(LKV, LQ)
                              head_dim::Int;
                              q_chunk_size = 128
                              ) where {T}
    y = querychunked_sdpa(xq, xk, xv, mask, head_dim, q_chunk_size=q_chunk_size)
    function sdpa_pullback(ȳ)
        k_len   = size(xk, 2)
        q_len   = size(xq, 2)
        kv_len  = size(xv, 2)
        nbatch  = size(xq, 3)
        q_chunk_size = min(q_chunk_size, q_len)
        α = sqrt(T(head_dim))
        
        xq̄, xk̄, xv̄ = similar(xq), fill!(similar(xk), 0), fill!(similar(xv), 0)
        Achunk = similar(xq, kv_len, q_chunk_size, nbatch)
        Āchunk = similar(xq, kv_len, q_chunk_size, nbatch)
        dMchunk = similar(xq, kv_len, q_chunk_size, nbatch)
        qk_chunk = similar(xq, kv_len, q_chunk_size, nbatch)
        qstart = 1
        while qstart <= q_len
            q_batch = min(q_chunk_size, q_len - qstart + 1)
            qinds = qstart:qstart+q_batch-1
            ȳview = view(ȳ,:,qinds,:)
            qk_chunkview = view(qk_chunk,:,1:q_batch,:)
            batched_mul!(qk_chunkview,batched_transpose(xk), view(xq, :, qinds, :), 1/α)
            Achunk[:,1:q_batch,:] .= softmax((qk_chunkview .+ view(mask,:,qinds)); dims=1)
            batched_mul!(xv̄, ȳview, batched_transpose(view(Achunk,:,1:q_batch,:)), one(T), one(T))
            Āchunkview = view(Āchunk,:,1:q_batch,:)
            batched_mul!(Āchunkview, batched_transpose(xv), ȳview)
            Achunkview = view(Achunk,:,1:q_batch,:)
            dMchunk[:,1:q_batch,:] .= (Achunkview .* (Āchunkview .- (sum(Achunkview .* Āchunkview, dims=1)))) ./ α #(LKV, LQ, HB)
            dMchunkview = view(dMchunk,:,1:q_batch,:)
            batched_mul!(xk̄, view(xq,:,qinds,:), batched_transpose(dMchunkview), one(T), one(T)) #(LKV, D, HB)
            batched_mul!(view(xq̄,:,qinds,:),xk, dMchunkview) #(D, LQ, HB)
            qstart += q_batch
        end
        return NoTangent(), xq̄, xk̄, xv̄, NoTangent(), NoTangent()
    end
    return y, sdpa_pullback
end
=#
