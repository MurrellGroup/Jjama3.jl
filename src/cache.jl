no_cache(k, v) = (k, v)
Base.position(::typeof(no_cache)) = 0

struct KVCache{T,A<:AbstractArray{T}}
    k::A
    v::A
    pos::Ref{Int}

    function KVCache(k::A, v::A, pos=0) where {T,A<:AbstractArray{T,4}}
        size(k) == size(v) || throw(DimensionMismatch("k and v must have the same size"))
        pos = pos isa Ref ? pos : Ref(pos)
        return new{T,A}(k, v, pos)
    end
end

Base.position(cache::KVCache) = cache.pos[]
position!(cache::KVCache, new_pos::Int) = cache.pos[] = new_pos

Base.size(cache::KVCache, args...) = size(cache.k, args...)
Base.length(cache::KVCache) = size(cache, 2)
Base.getindex(cache::KVCache, i) = KVCache(selectdim(cache.k, 2, i), selectdim(cache.v, 2, i))
current_sequence(cache::KVCache) = cache[1:position(cache)]
batch_size(cache::KVCache) = size(cache, 4)

function Base.show(io::IO, ::MIME"text/plain", cache::KVCache)
    println(io, typeof(cache), ':')
    println(io, "  size: $(size(cache.k))")
    println(io, "  position: $(position(cache)) / $(length(cache))")
    print(io, "  batches: $(batch_size(cache))")
end

function kv_cache(layer::Attention, len::Int, batch::Int=1)
    k = similar(layer.wq.weight, layer.head_dim, len, layer.n_kv_heads, batch) .= 0
    v = similar(layer.wv.weight, layer.head_dim, len, layer.n_kv_heads, batch) .= 0
    return KVCache(k, v)
end

no_kv_cache(layer::Attention) = no_cache

function extend(cache::KVCache, new_len::Int)
    head_dim, len, kv_heads, batch = size(cache.k)
    @assert new_len > len
    k = similar(cache.k, head_dim, new_len, kv_heads, batch) .= 0
    v = similar(cache.v, head_dim, new_len, kv_heads, batch) .= 0
    k[:, 1:len, :, :] .= cache.k
    v[:, 1:len, :, :] .= cache.v
    return KVCache(k, v, cache.pos)
end

function (cache::KVCache)(k::AbstractArray, v::AbstractArray)
    cache.k[:, position(cache) .+ axes(k, 2), :, :] .= k
    cache.v[:, position(cache) .+ axes(v, 2), :, :] .= v
    position!(cache, position(cache) + size(k, 2))
    return @views cache.k[:, 1:position(cache), :, :], cache.v[:, 1:position(cache), :, :]
end

function unrope!(cache::KVCache, rope::RoPE)
    rope = rope[1:size(cache.k, 2)]
    cache.k .= unrope(rope, cache.k)
    cache.v .= unrope(rope, cache.v)
    return cache
end

function rerope!(cache::KVCache, rope::RoPE)
    rope = rope[1:size(cache.k, 2)]
    cache.k .= rope(cache.k)
    cache.v .= rope(cache.v)
    return cache
end

function Base.append!(cache1::KVCache, cache2::KVCache)
    position(cache1) + position(cache2) > length(cache1) && throw(DimensionMismatch("appending cache2 to cache1 would exceed the sequence length of cache1"))
    cs = current_sequence(cache1)
    cache1.k[:, position(cache1) .+ axes(cs.k, 2), :, :] .= cs.k
    cache1.v[:, position(cache1) .+ axes(cs.v, 2), :, :] .= cs.v
    position!(cache1, position(cache1) + size(cache2.k, 2))
    return cache1
end

function scrape(cache::KVCache)
    cs = current_sequence(cache)
    return (; k=collect(cs.k), v=collect(cs.v))
end


struct KVCacheStack{C}
    caches::Vector{C}
end

Base.iterate(cache::KVCacheStack, state...) = iterate(cache.caches, state...)

function Base.show(io::IO, ::MIME"text/plain", cache::KVCacheStack)
    println(io, typeof(cache), ':')
    println(io, "  caches: $(length(cache.caches))")
    println(io, "  position: $(position(cache))")
end

extend(cache::KVCacheStack, new_len::Int) = KVCacheStack([extend(c, new_len) for c in cache.caches])

Base.position(cache::KVCacheStack) = only(unique(position.(cache.caches)))
position!(cache::KVCacheStack, new_pos::Int) = only(unique(position!.(cache.caches, new_pos)))

function kv_cache(model::Transformer, args...; kws...)
    return KVCacheStack([kv_cache(layer.attention, args...; kws...) for layer in model.layers])
end

no_kv_cache(model::Transformer) = KVCacheStack([no_kv_cache(layer.attention) for layer in model.layers])

function unrope!(caches::KVCacheStack, rope::RoPE)
    for cache in caches
        unrope!(cache, rope)
    end
    return caches
end

function rerope!(caches::KVCacheStack, rope::RoPE)
    for cache in caches
        rerope!(cache, rope)
    end
    return caches
end

function append!(caches1::KVCacheStack, caches2::KVCacheStack)
    for (cache1, cache2) in zip(caches1.caches, caches2.caches)
        append!(cache1, cache2)
    end
    return caches1
end

function scrape(caches::KVCacheStack)
    scraped_caches = [scrape(cache) for cache in caches]
    return (; k=[c.k for c in scraped_caches], v=[c.v for c in scraped_caches])
end
