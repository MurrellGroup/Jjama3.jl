module Jjama3

using ConcreteStructs
using Einops
using Flux
using SafeTensors
using LinearAlgebra
using NNlib
using LogitSamplers
using LowRankLayers
using ChainRulesCore

const causal_mask = Val(:causal_mask)

include("layers.jl")
export FeedForward
export RMSNorm
export RoPE
export Attention
export TransformerBlock
export Transformer
export unrope
export rerope_cache!
export scrape_cache
export append_cache!

include("cache.jl")
export kv_cache
export no_kv_cache
export position!

include("sdpa/sdpa.jl")
export sdpa
export flash_attention

include("model.jl")
export forward_loss
export forward_inference
export loss

include("sampling.jl")
export top_pk_sampler
export argmax_sampler
export top_nσ_sampler
export min_p_sampler
export generate

include("utils.jl")
export encode
export decode
export pad_and_batch
export structured_choice

include("models/models.jl")
export export_model

end
