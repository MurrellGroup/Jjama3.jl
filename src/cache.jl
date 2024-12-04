mutable struct KVCache{T,A<:AbstractArray{T,4}}
    cache_k::A
    cache_v::A
end

Flux.@layer KVCache

head_dim(cache::KVCache) = size(cache.cache_k, 1)
seq_length(cache::KVCache) = size(cache.cache_k, 2)
n_kv_heads(cache::KVCache) = size(cache.cache_k, 3)
batch_size(cache::KVCache) = size(cache.cache_k, 4)

function KVCache(T; head_dim, seq_length=0, n_kv_heads, batch_size=1)
    cache_k = zeros(T, head_dim, seq_length, n_kv_heads, batch_size)
    cache_v = zeros(T, head_dim, seq_length, n_kv_heads, batch_size)
    return KVCache(cache_k, cache_v)
end

function config!(cache::KVCache; seq_length=seq_length(cache), batch_size=batch_size(cache))
    cache.cache_k = similar(cache.cache_k, head_dim(cache), seq_length, n_kv_heads(cache), batch_size) .= 0
    cache.cache_v = similar(cache.cache_v, head_dim(cache), seq_length, n_kv_heads(cache), batch_size) .= 0
end

clear!(cache::KVCache) = config!(cache, seq_length=0)

function update!(cache::KVCache, start_pos::Int, xk::AbstractArray, xv::AbstractArray)
    if iszero(seq_length(cache))
        println("fuck")
        return xk, xv
    else
        seqlen = size(xk, 2)
        cache.cache_k[:, start_pos+1:start_pos+seqlen, :, :] .= xk
        cache.cache_v[:, start_pos+1:start_pos+seqlen, :, :] .= xv
        return cache.cache_k[:, 1:start_pos+seqlen, :, :],
            cache.cache_v[:, 1:start_pos+seqlen, :, :]
    end
end
