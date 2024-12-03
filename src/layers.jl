const AnyDense = Union{Dense, LoRADense}

struct KVCache{T}
    cache_k::AbstractArray{T,4}  # (head_dim, seq_len, n_kv_heads, batch)
    cache_v::AbstractArray{T,4} 
end

function KVCache(T, batch_size::Int, seq_length::Int, n_kv_heads::Int, head_dim::Int; device = identity)
    cache_k = zeros(T, head_dim, seq_length, n_kv_heads, batch_size) |> device
    cache_v = zeros(T, head_dim, seq_length, n_kv_heads, batch_size) |> device
    KVCache(cache_k, cache_v)
end


struct FeedForward{W<:AnyDense}
    w1::W
    w2::W
    w3::W
end

function FeedForward(dim::Int, ff_hidden_dim::Int)
    FeedForward(
        Dense(dim => ff_hidden_dim, bias=false),
        Dense(ff_hidden_dim => dim, bias=false),
        Dense(dim => ff_hidden_dim, bias=false)
    )
end

(ff::FeedForward)(x) = ff.w2(Flux.swish(ff.w1(x)) .* ff.w3(x))

Flux.@layer :expand FeedForward


struct RMSNorm{T,W<:AbstractVector{T}}
    weight::W
    eps::T
end

RMSNorm(dim::Int; eps::T=1f-5) where {T} = RMSNorm(ones(T, dim), eps)

function (norm::RMSNorm)(x)
    rms = sqrt.(sum(abs2, x, dims=1) ./ size(x,1) .+ norm.eps)
    return x .* (norm.weight ./ rms)
end

Flux.@layer RMSNorm


struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end

Base.getindex(rope::RoPE, i) = RoPE(selectdim(rope.cos, 2, i), selectdim(rope.sin, 2, i))

function apply_scaling!(freqs::AbstractVector; scale_factor=8)
    #Hard-coded - I should move these to the main model struct and grab them from the config.
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    ###
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    for (i, freq) in enumerate(freqs)
        wavelen = 2π / freq
        if wavelen > low_freq_wavelen
            freqs[i] = freq / scale_factor
        elseif wavelen > high_freq_wavelen
            @assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / 
                    (high_freq_factor - low_freq_factor)
            freqs[i] = (1 - smooth) * freq / scale_factor + smooth * freq
        end
    end
    return freqs
end

function RoPE(
    dim::Int, end_pos::Int; 
    theta::T=10000f0, use_scaled=true, scale_factor=8,
) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    use_scaled && apply_scaling!(freqs; scale_factor)
    freqs_complex = cis.(T.(0:end_pos-1) * freqs')
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos, 1, 1))
    sin = reshape(sin, (dim÷2, end_pos, 1, 1))
    return RoPE(cos, sin)
end

#Note about Huggingface weights and rotary embeddings: https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
#Use this one if you're using the Hugging Face weights.
function (rope::RoPE)(x)
    head_dim = size(x, 1)
    x1 = @view x[1:head_dim÷2, :, :, :]
    x2 = @view x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end


mutable struct Attention{Q,K,V,O}
    wq::Q
    wk::K
    wv::V
    wo::O
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    n_rep::Int
    #cache::Union{Nothing, KVCache}
end

function Attention(dim::Int, n_heads::Int, n_kv_heads=n_heads; qkv_bias=false)
    head_dim = dim ÷ n_heads
    n_rep = n_heads ÷ n_kv_heads
    Attention(
        Dense(dim => n_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(n_heads * head_dim => dim, bias=false),
        n_heads,
        n_kv_heads,
        head_dim,
        n_rep,
        #nothing
    )
end

function (attn::Attention)(x::AbstractArray{T}, start_pos::Int, rope::RoPE, mask=false) where T
    dim, seqlen, batch = size(x)

    xq = attn.wq(x)
    xk = attn.wk(x)
    xv = attn.wv(x)

    xq = reshape(xq, (attn.head_dim, attn.n_heads, seqlen, batch))
    xk = reshape(xk, (attn.head_dim, attn.n_kv_heads, seqlen, batch))
    xv = reshape(xv, (attn.head_dim, attn.n_kv_heads, seqlen, batch))

    xq = permutedims(xq, (1,3,2,4))
    xk = permutedims(xk, (1,3,2,4))
    xv = permutedims(xv, (1,3,2,4))

    xq_rope = rope(xq)
    xk_rope = rope(xk)

    #@trace if !isnothing(attn.cache)
    #    xk_rope, xv = update_kv_cache!(attn.cache, start_pos, xk_rope, xv)
    #end
    xk_rope = repeat_kv(xk_rope, attn.n_rep)
    xv      = repeat_kv(xv, attn.n_rep)
    
    xq_for_attn = reshape(xq_rope, attn.head_dim, :,  attn.n_heads * batch)
    xk_for_attn = reshape(xk_rope, attn.head_dim, :, attn.n_heads * batch)
    xv_for_attn = reshape(xv, attn.head_dim, :, attn.n_heads * batch)

    scores = batched_mul(batched_transpose(xk_for_attn), xq_for_attn) / sqrt(T(attn.head_dim))
    scores .+= mask
    sm_scores = softmax(scores; dims=1)
    output = batched_mul(xv_for_attn, sm_scores)
    e_output = reshape(output, (attn.head_dim, seqlen, attn.n_heads, batch))
    p_output = permutedims(e_output, (1,3,2,4)) 
    
    r_output = reshape(p_output, (attn.head_dim * attn.n_heads, seqlen, batch))
    proj = attn.wo(r_output)
    return proj
end

Flux.@layer :expand Attention trainable=(wq,wv)


struct TransformerBlock{A<:Attention,F<:FeedForward,AN<:RMSNorm,FN<:RMSNorm}
    attention::A
    feed_forward::F
    attention_norm::AN
    ffn_norm::FN
end

function TransformerBlock(dim::Int, n_heads::Int, n_kv_heads::Int=n_heads, ff_hidden_dim = 4 * dim;
                         norm_eps=1f-5, qkv_bias=false)
    TransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        FeedForward(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps)
    )
end

function (block::TransformerBlock)(x, start_pos, rope, mask=nothing)
    h = x + block.attention(block.attention_norm(x), start_pos, rope, mask)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

Flux.@layer TransformerBlock trainable=(attention,)


struct Transformer{E<:Flux.Embedding,B<:AbstractVector{<:TransformerBlock},N<:RMSNorm,O<:Dense,R<:RoPE}
    tok_embeddings::E
    layers::B
    norm::N
    output::O
    rope::R
end

function Transformer(
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps::T=1f-5,
    qkv_bias=false,
    rope_theta::T=500000f0,
    use_scaled_rope=false,
    scale_factor=8,
) where T
    tok_embeddings = Flux.Embedding(vocab_size => dim)
    layers = [TransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps=norm_eps, qkv_bias=qkv_bias) for _ in 1:n_layers]
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    rope = RoPE(dim ÷ n_heads, max_seq_len * 2; theta=rope_theta, use_scaled=use_scaled_rope, scale_factor=scale_factor)
    Transformer(tok_embeddings, layers, norm, output, rope)
end

Flux.@layer :expand Transformer trainable=(layers,)
