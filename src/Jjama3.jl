module Jjama3

using Flux
using SafeTensors
using Distributions
using LinearAlgebra
using StatsBase
using NNlib
using LogitSamplers
using LowRankLayers

using HuggingFaceTokenizers: HuggingFaceTokenizers, Tokenizer

const tokenizer_from_repo = HuggingFaceTokenizers.from_pretrained
const tokenizer_from_file = HuggingFaceTokenizers.from_file

include("cache.jl")
export KVCache

include("layers.jl")
export FeedForward
export RMSNorm
export RoPE
export Attention
export TransformerBlock
export Transformer
export unrope

include("model.jl")
export forward_loss
export forward_inference

include("sampling.jl")
export top_pk_sampler
export argmax_sampler
export top_nσ_sampler
export min_p_sampler
export generate
export tokenizer_from_repo
export tokenizer_from_file
export Tokenizer

include("utils.jl")
export encode
export decode
export load_llama321B_from_safetensors
export load_llama3_from_safetensors
export llama3_instruct_prompt
export llama3_assistant_prompt
export smollm2_instruct_prompt
export smollm2_assistant_prompt
export structured_choice
export pad_and_batch
export export_model

end
