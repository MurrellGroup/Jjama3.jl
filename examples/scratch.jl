#Pkg.add(["Flux", "JSON3", "UnicodePlots", "StatsBase"])
using Jjama3, Flux, StatsBase, UnicodePlots

#Init a tiny model
model = Transformer(
    22,             # vocab_size
    16*8,           # dim
    12,             # n_layers
    8,              # n_heads
    4,              # n_kv_heads
    8192,           # max_seq_len
    16*10,          # ff_hidden_dim
)

#Make everything except the RoPE trainable
Jjama3.Flux.@layer Jjama3.Transformer trainable=(tok_embeddings, layers, norm, output)
Jjama3.Flux.@layer Jjama3.Attention trainable=(wq, wk, wv, wo)
Jjama3.Flux.@layer Jjama3.TransformerBlock trainable=(attention, feed_forward, attention_norm, ffn_norm)

#Set up trivial tokenizer
AAs = collect(">ACDEFGHIKLMNPQRSTVWY.")

#Read data, remove X-containing sequences, and adding start and end tokens
data = readlines("abs.txt")
data = [">"*d*"." for d in data if !(occursin("X", d))]

#Train the model
lr = 0.001f0
opt_state = Flux.setup(AdamW(lr), model)
losses = Float32[]
for i in 1:2001
    #Prep random batch
    train_toks = pad_and_batch(encode.((AAs, ), data[sample(1:length(data), 10, replace=false)]), 22);
    #Compute loss and gradients
    loss, grads = Flux.withgradient(model) do m
        forward_loss(m, train_toks[1:end-1,:], train_toks[2:end,:])
    end
    #Update weights
    Flux.update!(opt_state, model, grads[1])
    #Monitor
    push!(losses, loss)
    println(i, " ", loss)
    #Monitor sampling
    if mod(i, 100) == 1
        generate(model, encode(AAs, ">"),
                max_new_tokens=500,
                tokenizer_for_printing=AAs,
                end_token = 22, sampler = top_pk_sampler(p = 1.0f0, k = 22))
        println()
        display(lineplot(losses, width = 150, height = 30))
    end
    #Linear learning rate cooldown
    if i > 1500
        lr = max(lr - 0.001f0/(2000-1500), 0.0000001f0)
        Flux.adjust!(opt_state, lr)
    end
end

#Test sampling
for i in 1:10
    println(">", i)
    generate(model, encode(AAs, ">"),
                max_new_tokens=500,
                tokenizer_for_printing=AAs,
                end_token = 22, sampler = top_pk_sampler(p = 1.0f0, k = 22))
    println()
end

#Exporting the model
export_model(model, "tinyabllama.safetensors", type_convert = x -> Jjama3.SafeTensors.BFloat16.(x))

#Saving a config so that it loads correctly using the Jjama3 loader
using JSON3
config = Dict()
config[:model_type] = "llama"    
config[:vocab_size]= 22
config[:hidden_size] = 16*8
config[:num_hidden_layers] = 12
config[:num_attention_heads] = 8
config[:num_key_value_heads] = 4
config[:max_position_embeddings] = 8192
config[:intermediate_size] = 16*10
config[:rms_norm_eps] = 1f-8
config[:rope_theta] = 500000f0
config[:tie_word_embeddings] = false
open("tinyabllama_config.json", "w") do f
    JSON3.pretty(f, JSON3.write(config))
    println(f)
end

#Load a trained model and test it
config = JSON3.read(read("tinyabllama_config.json", String))
model_weight_paths = ["tinyabllama.safetensors"]
model = load_llama3_from_safetensors(model_weight_paths, config)
@assert generate(model, encode(AAs, ">"), end_token = 22) == [1, 15, 19, 15, 11, 19, 15, 17, 7, 2, 5, 19, 10, 10, 14, 7, 2, 17, 19, 10, 19, 17, 3, 10, 2, 17, 7, 21, 18, 6, 18, 17, 21, 7, 9, 17, 20, 19, 16, 15, 2, 14, 7, 15, 7, 11, 5, 20, 12, 7, 20, 9, 17, 2, 21, 13, 7, 13, 18, 13, 21, 2, 15, 10, 11, 15, 7, 16, 19, 18, 12, 18, 18, 4, 18, 17, 18, 17, 18, 2, 21, 12, 5, 11, 16, 17, 11, 16, 17, 4, 4, 18, 2, 19, 21, 21, 3, 2, 16, 4, 16]