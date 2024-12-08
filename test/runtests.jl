using Jjama3
using Test
using JSON3

@testset "Jjama3.jl" begin

    @testset "Amino Acid Model" begin
        url_branch = "https://raw.githubusercontent.com/MurrellGroup/Jjama3.jl/aminoacid-model/"
        config_path = download(url_branch * "tinyabllama_config.json")
        model_path = download(url_branch * "tinyabllama.safetensors")
        config = JSON3.read(read(config_path, String))
        model = load_llama3_from_safetensors([model_path], config)
        AAs = collect(">ACDEFGHIKLMNPQRSTVWY.")
        @test generate(model, encode(AAs, ">"), end_token = 22) == [1, 15, 19, 15, 11, 19, 15, 17, 7, 2, 5, 19, 10, 10, 14, 7, 2, 17, 19, 10, 19, 17, 3, 10, 2, 17, 7, 21, 18, 6, 18, 17, 21, 7, 9, 17, 20, 19, 16, 15, 2, 14, 7, 15, 7, 11, 5, 20, 12, 7, 20, 9, 17, 2, 21, 13, 7, 13, 18, 13, 21, 2, 15, 10, 11, 15, 7, 16, 19, 18, 12, 18, 18, 4, 18, 17, 18, 17, 18, 2, 21, 12, 5, 11, 16, 17, 11, 16, 17, 4, 4, 18, 2, 19, 21, 21, 3, 2, 16, 4, 16]
    end

end
