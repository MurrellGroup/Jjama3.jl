@concrete struct FeedForward
    w1
    w2
    w3
end

Flux.@layer FeedForward

function FeedForward(dim::Int, ff_hidden_dim::Int)
    FeedForward(
        Dense(dim => ff_hidden_dim, bias=false),
        Dense(ff_hidden_dim => dim, bias=false),
        Dense(dim => ff_hidden_dim, bias=false)
    )
end

(ff::FeedForward)(x) = ff.w2(Flux.swish(ff.w1(x)) .* ff.w3(x))


@concrete struct RMSNorm
    weight
    eps
end

Flux.@layer RMSNorm

RMSNorm(dim::Int; eps=1f-5) = RMSNorm(ones(typeof(eps), dim), eps)

function (norm::RMSNorm)(x::AbstractArray{T}) where T
    rms = sqrt.(sum(abs2, x, dims=1) ./ size(x, 1) .+ norm.eps)
    return x .* (norm.weight ./ rms)
end


@concrete struct RoPE
    cos
    sin
end

Flux.@layer RoPE trainable=()

Base.getindex(rope::RoPE, i) = @views RoPE(rope.cos[:,i,:,:], rope.sin[:,i,:,:])

function apply_scaling!(freqs::AbstractVector; scale_factor=8)
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
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
    theta::T=10000f0, use_scaled=true, scale_factor=8, start_pos=0
) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    use_scaled && apply_scaling!(freqs; scale_factor)
    freqs_complex = cis.(T.(start_pos:end_pos-1) * freqs')
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos - start_pos, 1, 1))
    sin = reshape(sin, (dim÷2, end_pos - start_pos, 1, 1))
    return RoPE(cos, sin)
end

# Note about Huggingface weights and rotary embeddings:
# https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
# Use this one if you're using the Hugging Face weights.
function (rope::RoPE)(x)
    n = size(x, 1)
    k = n ÷ 2
    x1 = selectdim(x, 1, 1:k)
    x2 = selectdim(x, 1, 1+k:n)
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end

function unrope(rope, x)
    n = size(x, 1)
    k = n ÷ 2
    x1 = selectdim(x, 1, 1:k)
    x2 = selectdim(x, 1, 1+k:n)
    return vcat(  
        x1 .* rope.cos .+ x2 .* rope.sin,
        x2 .* rope.cos .- x1 .* rope.sin
    )
end

@concrete struct Attention
    wq; wk; wv; wo
    q_norm; k_norm
    in_dim::Int
    head_dim::Int
    n_heads::Int
    n_kv_heads::Int
end

Flux.@layer Attention

function Attention(
    in_dim::Int, n_heads::Int, n_kv_heads::Int=n_heads;
    head_dim = in_dim ÷ n_heads, qk_norm=false, qkv_bias=false,
)
    return Attention(
        Dense(in_dim => n_heads * head_dim, bias=qkv_bias),
        Dense(in_dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(in_dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(n_heads * head_dim => in_dim, bias=false),
        qk_norm ? RMSNorm(head_dim) : identity,
        qk_norm ? RMSNorm(head_dim) : identity,
        head_dim, head_dim, n_heads, n_kv_heads)
end

function (layer::Attention)(xq::AbstractArray{T,3}, xk::AbstractArray{T,3}=xq; rope=identity, cache=no_cache, sdpa=sdpa, mask=false) where T
    q, k, v = layer.wq(xq), layer.wk(xk), layer.wv(xk)
    q, k, v = rearrange.((q, k, v), einops"(d h) l ... -> d l h ..."; d=layer.head_dim)
    q, k = layer.q_norm(q), layer.k_norm(k)
    q, k = rope(q), rope(k)
    k, v = cache(k, v)
    k, v = repeat.((k, v), einops"d l h ... -> d l (r h) ..."; r=layer.n_heads÷layer.n_kv_heads)
    x = sdpa(q, k, v; mask)
    x = rearrange(x, einops"d l h ... -> (d h) l ...")
    return layer.wo(x)
end

function (layer::Attention)(xs::AbstractArray...; kws...)
    xs = rearrange.(xs, einops"dim len ... -> dim len (...)")
    return layer(xs...; kws...)
end

@concrete struct TransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
end

Flux.@layer TransformerBlock

function TransformerBlock(
    in_dim::Int, n_heads::Int, n_kv_heads::Int=n_heads, ff_hidden_dim::Int=4*in_dim;
    norm_eps=1f-5, head_dim=in_dim ÷ n_heads, kws...
)
    TransformerBlock(
        Attention(in_dim, n_heads, n_kv_heads; head_dim, kws...),
        FeedForward(in_dim, ff_hidden_dim),
        RMSNorm(in_dim, eps=norm_eps),
        RMSNorm(in_dim, eps=norm_eps)
    )
end

function (block::TransformerBlock)(x; kws...)
    h = x + block.attention(block.attention_norm(x); kws...)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

@concrete struct Transformer
    embeddings
    layers
    norm
    output
    rope
end

Flux.@layer Transformer

function Transformer(
    vocab_size::Int, dim::Int, n_layers::Int, n_heads::Int, 
    n_kv_heads::Int, max_seq_len::Int, ff_hidden_dim::Int;
    norm_eps = 1f-5,
    rope_settings = (theta = 500000f0, use_scaled = false, scale_factor = 8),
    head_dim = dim ÷ n_heads,
    kws...
)
    embeddings = Embedding(vocab_size => dim)
    layers = Tuple(TransformerBlock(dim, n_heads, n_kv_heads, ff_hidden_dim; norm_eps, head_dim, kws...) for _ in 1:n_layers)
    norm = RMSNorm(dim, eps=norm_eps)
    output = Dense(dim => vocab_size, bias=false)
    rope = RoPE(head_dim, max_seq_len * 2; rope_settings...)
    Transformer(embeddings, layers, norm, output, rope)
end
