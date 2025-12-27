using NNop: NNop
using GPUArraysCore: AbstractGPUArray

function flash_attention(
    q::AbstractGPUArray, k::AbstractGPUArray, v::AbstractGPUArray;
    mask=false,
)
    causal = mask === causal_mask
    return NNop.flash_attention(q, k, v; causal)
end
