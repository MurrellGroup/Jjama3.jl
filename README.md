# Jjama3 - Hackable Llama3.1 and Llama3.2 (text) in Julia

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Jjama3.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Jjama3.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Jjama3.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Jjama3.jl)

## Latest

- Now with support for the Qwen 2.5 (eg. [base](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e), [Qwen2.5-Coder](https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f), amd [Qwen2.5-Math](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e)).

## Installation


We've split this into a few (unregistered) packages, so you'll need to add them all, and you need JSON3 for loading the configs:
```
] add JSON3
] add https://github.com/MurrellGroup/HuggingFaceTokenizers.jl
] add https://github.com/MurrellGroup/LowRankLayers.jl
] add https://github.com/MurrellGroup/LogitSamplers.jl
] add https://github.com/MurrellGroup/Jjama3.jl
```

## Quickstart

Download a Llama3 model `config.json`, `tokenizer.json`, and model safetensor weights from Hugging Face. Eg. [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/main). Note: Hugging Face Llama3 models use a different RoPE convention to the original Meta implementation, and their weights have been permuted. This package works with the Huggingface convention, so if you load from the original Meta-Llama weights from a different source you'll need to do something horrible.

```julia
using JSON3, Jjama3

config = JSON3.read(read("SmolLM2-360M-Instruct/config.json", String))
model = load_llama3_from_safetensors("SmolLM2-360M-Instruct/model.safetensors", config)
tkn = tokenizer_from_file(Tokenizer, "SmolLM2-360M-Instruct/tokenizer.json")

prompt = smollm2_assistant_prompt(tkn,"Tell me the two worst things about Python.")
generate(model, prompt,
        max_new_tokens=500,
        tokenizer_for_printing=tkn,
        end_token = encode(tkn, "<|im_end|>")[end]);
```

## Capability

- Works with Llama3.1 and Llama3.2 models (also including SmolLM2), loaded from Huggingface safetensors.
- Sampling accelerated with KV caching.
- RoPE scaling (for exceeding the model's max training-time context length) is implemented, but likely incorrect with KV cache. Be careful if you're using with really long sequences.
- Imported models are trainable (with Flux), including with low-rank (ie. LoRA) finetuning.
- Sampling, training, etc compatible with CUDA, where everything is much faster.
- Metal acceleration for forward_inference, forward_loss, and sampling. Gradients (with Zygote) fail. Sampling works, but is slower with Metal than with CPU.


## Samplers

The transformer emits "logits" which control the probability of the next token. A sampler takes these logits and converts them into a probability distribution over the vocabulary, and then samples from this distribution. There are [a few samplers available](https://github.com/MurrellGroup/LogitSamplers.jl), including argmax, top-p, top-k, min-p, and top-nσ. These can substantially affect the output of the model.

```julia
prompt = smollm2_assistant_prompt(tkn,"Tell me the two worst things about Python.");

generate(model, prompt,
        max_new_tokens=500,
        tokenizer_for_printing=tkn,
        end_token = encode(tkn, "<|im_end|>")[end],
        sampler = top_nσ_sampler());
```

## Structured Sampling

You can pass in a custom sampler that places additional constraints on the sampling process. As an example, `structured_choice` is a sampler that always selects from a set of predefined options:

```julia
question = "In a Bayesian model, what do we call the probability distribution of parameters given the data?"
choices = ["Prior",
           "Likelihood",
           "Marginal Likelihood",
           "Evidence",
           "Posterior"]

vocab = [decode(tkn, [i], skip_special_tokens = false) for i in 1:size(model.output.weight,1)]
eos = encode(tkn, "<|im_end|>")[end]

sysprompt = "You are an expert in Statistics and Probability Theory who answers questions in as few words as possible."
prompt = smollm2_instruct_prompt(tkn, sysprompt, question)

generate(model, prompt,
        max_new_tokens=100,
        tokenizer_for_printing=tkn,
        end_token = eos,
        sampler = structured_choice(choices, vocab, eos));
```

This strategy can be extended to force the model outputs to follow specific formats.

## Finetuning

Often we want to adjust model parameters to better fit our specific use case, by further training the model on a new dataset. This can be done on all the model weights, but we also provide low-rank (via LoRA) finetuning.

```julia
using Jjama3, JSON3, Flux

config = JSON3.read(read("SmolLM2-360M-Instruct/config.json", String))
tkn = tokenizer_from_file(Tokenizer, "SmolLM2-360M-Instruct/tokenizer.json")
eos = encode(tkn, "<|im_end|>")[end]

#Add LoRA to Q and V matrices when loading the model
model = load_llama3_from_safetensors("SmolLM2-360M-Instruct/model.safetensors", config,
                                        add_lora_to = [:Q, :V], lora_dim = 64)

#See how the model answers before finetuning
prompt = smollm2_assistant_prompt(tkn, "What language is the best for deep learning?");
generate(model, prompt, max_new_tokens=50, tokenizer_for_printing=tkn, end_token = eos);

#Set up a single, very silly, training example to finetune on
ugh = "Ugh, bruh, what a stupid question.<|im_end|>"
trainsample = decode(tkn, prompt, skip_special_tokens = false) * ugh;
train_toks = encode(tkn, trainsample);

#Set up the optimizer
opt_state = Flux.setup(AdamW(0.001f0), model);

#Train for 5 steps, monitoring the model's output as it tunes
for i in 1:5
    grads = Flux.gradient(model) do m
        forward_loss(m, train_toks[1:end-1,:], train_toks[2:end,:])
    end
    Flux.update!(opt_state, model, grads[1])
    println(i)
    generate(model, prompt,
            max_new_tokens=50,
            tokenizer_for_printing=tkn,
            end_token = eos)
    println()
end

#Ask the model an unrelated question to see how stupid we've made the model. Try this a few times.
prompt = smollm2_assistant_prompt(tkn, "Explain how tides work?");
generate(model, prompt,
        max_new_tokens=500,
        tokenizer_for_printing=tkn,
        end_token = eos,
        sampler = top_nσ_sampler());
```

## CUDA GPU

```julia
using CUDA, Flux, JSON3, Jjama3
```

For sampling, you can pass `device = gpu` to the `generate` function:

```julia
#Put the model on the GPU
model = gpu(model)

prompt = smollm2_assistant_prompt(tkn,"Tell me the two worst things about Python.")
generate(model, prompt,
        max_new_tokens=500,
        tokenizer_for_printing=tkn,
        end_token = encode(tkn, "<|im_end|>")[end],
        device = gpu); #Note the device keyword
```

And if you're training, the data needs to be on the GPU:

```julia
model = gpu(model)

train_toks = encode(tkn, "This is a test.")
gpu_train_toks = gpu(train_toks)

forward_loss(model, gpu_train_toks[1:end-1,:], gpu_train_toks[2:end,:])
```