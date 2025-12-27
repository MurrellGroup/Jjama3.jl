module HuggingFaceTokenizersExt

using Jjama3
using HuggingFaceTokenizers

Jjama3.encode(tkn::HuggingFaceTokenizers.Tokenizer, str; kws...) = HuggingFaceTokenizers.encode(tkn, str; kws...).ids .+ 1
Jjama3.decode(tkn::HuggingFaceTokenizers.Tokenizer, ids; kws...) = HuggingFaceTokenizers.decode(tkn, ids .- 1; kws...)

end
