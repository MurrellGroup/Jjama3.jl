module Jjama3

using Flux, SafeTensors, Distributions, LinearAlgebra, StatsBase, NNlib
using LogitSamplers, LowRankLayers
import HuggingFaceTokenizers


tokenizer_from_repo = HuggingFaceTokenizers.from_pretrained
tokenizer_from_file = HuggingFaceTokenizers.from_file
Tokenizer = HuggingFaceTokenizers.Tokenizer

top_pk_sampler = LogitSamplers.top_pk_sampler
argmax_sampler = LogitSamplers.argmax_sampler
min_p_sampler = LogitSamplers.min_p_sampler
top_nσ_sampler = LogitSamplers.top_nσ_sampler



include("layers.jl")
include("model.jl")
include("utils.jl")
include("sampling.jl")

export  load_llama321B_from_safetensors, 
        load_llama3_from_safetensors, 
        generate, 
        forward_loss, 
        forward_inference, 
        top_pk_sampler, 
        argmax_sampler,
        top_nσ_sampler,
        min_p_sampler,
        tokenizer_from_repo,
        tokenizer_from_file,
        encode,
        decode,
        Tokenizer,
        llama3_instruct_prompt,
        llama3_assistant_prompt,
        smollm2_instruct_prompt,
        smollm2_assistant_prompt,
        structured_choice

end