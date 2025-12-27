#Trivial Char tokenizers:
encode(chars::Vector{Char}, str::String) = [findfirst(==(c), chars) for c in str]
decode(chars::Vector{Char}, enc::Vector{Int}; skip_special_tokens=false) = String([chars[i] for i in enc])

#For training:
function pad_and_batch(seqs, pad_token)
    max_len = maximum(length.(seqs))
    padded = [vcat(s, fill(pad_token, max_len - length(s))) for s in seqs]
    cat(padded..., dims = 2)
end


"""
    sampler = structured_choice(choices, vocab::Vector{String}, end_token::Int; sampler = logits -> argmax_sampler(logits))

Return a function that can be passed into generate as a sampler, which will sample from the given choices. Handles the case where the choices are made up of multiple tokens.
`vocab` is an array of the tokens as strings, in their order in the tokenizer. `sampler` is a function that takes the logits (here including those masked with -Inf) and returns a sample from them. Defaults to argmax.

Example:
```julia
config = JSON3.read(read("SmolLM2-1.7B-Instruct/config.json", String))
model = load_llama3_from_safetensors("SmolLM2-1.7B-Instruct/model.safetensors", config)
tkn = tokenizer_from_file(Tokenizer, "SmolLM2-1.7B-Instruct/tokenizer.json")

question = "In a Bayesian model, what do we call the probability distribution of parameters given the data?"
choices = ["Prior", "Likelihood", "Marginal Likelihood", "Evidence", "Posterior"]

vocab = [decode(tkn, [i], skip_special_tokens = false) for i in 1:49152]
eos = encode(tkn, "<|im_end|>")[end]
prompt = smollm2_instruct_prompt(tkn, "You are an expert in Statistics and Probability Theory who answers questions in as few words as possible.",question)
generate(model, prompt, max_new_tokens=100, tokenizer_for_printing=tkn, end_token = eos, sampler = structured_choice(choices, vocab, eos));
```

If you want to run the model on the GPU, then you need to pass `device = gpu` to the `generate` function, and `device = cpu` to the `structured_choice` function.
"""
function structured_choice(choices::Vector{String}, vocab::Vector{String}, end_token::Int; sampler = logits -> argmax_sampler(logits), device = identity)
    remaining_choices = copy(choices)
    function choice_sampler(logits)
        logits = device(logits)
        if length(remaining_choices) == 0 || maximum(length.(remaining_choices)) == 0
            return end_token
        end
        mask = zeros(Bool, length(vocab))
        for i in 1:length(vocab)
            for choice in remaining_choices
                if startswith(choice, vocab[i])
                    mask[i] = true
                end
            end
        end
        logits[.!mask] .= -Inf
        next_token = sampler(logits)
        next_token_str = vocab[next_token]
        remaining_choices = [choice[length(next_token_str)+1:end] for choice in remaining_choices if startswith(choice, next_token_str)]
        return next_token
    end
    return choice_sampler
end
