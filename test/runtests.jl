using Jjama3
using Test
using JSON3
using Downloads

@testset "Jjama3.jl" begin

    @testset "Amino Acid Model" begin
        url_branch = "https://raw.githubusercontent.com/MurrellGroup/Jjama3.jl/aminoacid-model/"
        config_path = Downloads.download(url_branch * "tinyabllama_config.json")
        model_path = Downloads.download(url_branch * "tinyabllama.safetensors")
        config = JSON3.read(read(config_path, String))
        model = load_llama3_from_safetensors([model_path], config)
        AAs = collect(">ACDEFGHIKLMNPQRSTVWY.")
        @test generate(model, encode(AAs, ">"), end_token = 22) |> vec == [1, 15, 19, 15, 11, 19, 5, 17, 7, 7, 7, 19, 19, 15, 14, 7, 16, 17, 11, 16, 11, 17, 3, 2, 2, 17, 7, 6, 18, 6, 17, 17, 21, 7, 12, 8, 20, 19, 16, 15, 2, 14, 7, 10, 7, 11, 5, 20, 19, 2, 19, 9, 17, 21, 4, 7, 17, 13, 10, 21, 21, 2, 4, 17, 19, 10, 7, 16, 6, 18, 9, 17, 16, 4, 13, 17, 10, 13, 18, 11, 21, 11, 15, 12, 13, 17, 11, 16, 2, 5, 4, 18, 2, 19, 21, 21, 3, 2, 10, 4, 16]
        @test generate(model, encode(AAs, ">"), end_token = 22, sdpa = Jjama3.keychunked_sdpa) |> vec == [1, 15, 19, 15, 11, 19, 5, 17, 7, 7, 7, 19, 19, 15, 14, 7, 16, 17, 11, 16, 11, 17, 3, 2, 2, 17, 7, 6, 18, 6, 17, 17, 21, 7, 12, 8, 20, 19, 16, 15, 2, 14, 7, 10, 7, 11, 5, 20, 19, 2, 19, 9, 17, 21, 4, 7, 17, 13, 10, 21, 21, 2, 4, 17, 19, 10, 7, 16, 6, 18, 9, 17, 16, 4, 13, 17, 10, 13, 18, 11, 21, 11, 15, 12, 13, 17, 11, 16, 2, 5, 4, 18, 2, 19, 21, 21, 3, 2, 10, 4, 16]
        @test_broken generate(model, encode(AAs, ">"), end_token = 22, sdpa = Jjama3.querychunked_sdpa) |> vec == [1, 15, 19, 15, 11, 19, 5, 17, 7, 7, 7, 19, 19, 15, 14, 7, 16, 17, 11, 16, 11, 17, 3, 2, 2, 17, 7, 6, 18, 6, 17, 17, 21, 7, 12, 8, 20, 19, 16, 15, 2, 14, 7, 10, 7, 11, 5, 20, 19, 2, 19, 9, 17, 21, 4, 7, 17, 13, 10, 21, 21, 2, 4, 17, 19, 10, 7, 16, 6, 18, 9, 17, 16, 4, 13, 17, 10, 13, 18, 11, 21, 11, 15, 12, 13, 17, 11, 16, 2, 5, 4, 18, 2, 19, 21, 21, 3, 2, 10, 4, 16]
    end

end
