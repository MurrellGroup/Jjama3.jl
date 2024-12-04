mutable struct KVCache{T,A<:AbstractArray{T,4}}
    head_dim::Int
    n_kv_heads::Int
    seq_length::Int
    batch_size::Int
    cache_k::A
    cache_v::A
end

Flux.@layer KVCache

function KVCache(T; head_dim, seq_length=0, n_kv_heads, batch_size=1)
    cache_k = zeros(T, head_dim, seq_length, n_kv_heads, batch_size)
    cache_v = zeros(T, head_dim, seq_length, n_kv_heads, batch_size)
    return KVCache(head_dim, n_kv_heads, seq_length, batch_size, cache_k, cache_v)
end

function config!(cache::KVCache; seq_length=cache.seq_length, batch_size=cache.batch_size)
    cache.cache_k = similar(cache.cache_k, cache.head_dim, seq_length, cache.n_kv_heads, batch_size) .= 0
    cache.cache_v = similar(cache.cache_v, cache.head_dim, seq_length, cache.n_kv_heads, batch_size) .= 0
end

clear!(cache::KVCache) = config!(cache, seq_length=0)

function update!(cache::KVCache, start_pos::Int, xk::AbstractArray, xv::AbstractArray)
    #if iszero(cache.seq_length)
    #    return xk, xv
    #else
        seqlen = size(xk, 2)
        cache.cache_k[:, start_pos+1:start_pos+seqlen, :, :] .= xk
        cache.cache_v[:, start_pos+1:start_pos+seqlen, :, :] .= xv
        return cache.cache_k[:, 1:start_pos+seqlen, :, :],
            cache.cache_v[:, 1:start_pos+seqlen, :, :]
    #end
end
