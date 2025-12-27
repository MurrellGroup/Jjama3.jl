include("llama3.jl")
export llama3_instruct_prompt
export llama3_assistant_prompt
export load_llama3_from_safetensors

include("qwen3.jl")
export qwen3_instruct_prompt
export qwen3_assistant_prompt
export load_qwen3_from_safetensors

include("smollm2.jl")
export smollm2_instruct_prompt
export smollm2_assistant_prompt

function export_model(model, output_path; type_convert = identity)
    weights = Dict{String,AbstractArray}()
    weights["model.embed_tokens.weight"] = type_convert(model.embeddings.weight')
    weights["lm_head.weight"] = type_convert(model.output.weight)
    weights["model.norm.weight"] = type_convert(model.norm.weight)
    for (i, layer) in enumerate(model.layers)
        prefix = "model.layers.$(i-1)"
        weights["$prefix.self_attn.q_proj.weight"] = type_convert(layer.attention.wq.weight)
        weights["$prefix.self_attn.k_proj.weight"] = type_convert(layer.attention.wk.weight)
        weights["$prefix.self_attn.v_proj.weight"] = type_convert(layer.attention.wv.weight)
        weights["$prefix.self_attn.o_proj.weight"] = type_convert(layer.attention.wo.weight)
        if layer.attention.wq.bias
            weights["$prefix.self_attn.q_proj.bias"] = type_convert(layer.attention.wq.bias)
            weights["$prefix.self_attn.k_proj.bias"] = type_convert(layer.attention.wk.bias)
            weights["$prefix.self_attn.v_proj.bias"] = type_convert(layer.attention.wv.bias)
        end
        weights["$prefix.mlp.gate_proj.weight"] = type_convert(layer.feed_forward.w1.weight)
        weights["$prefix.mlp.down_proj.weight"] = type_convert(layer.feed_forward.w2.weight)
        weights["$prefix.mlp.up_proj.weight"] = type_convert(layer.feed_forward.w3.weight)
        weights["$prefix.input_layernorm.weight"] = type_convert(layer.attention_norm.weight)
        weights["$prefix.post_attention_layernorm.weight"] = type_convert(layer.ffn_norm.weight)
    end
    SafeTensors.serialize(output_path, weights)
    return nothing
end
