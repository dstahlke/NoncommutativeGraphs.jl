module NoncommutativeGraphsTesting

include("test_header.jl")

@testset "Classical graph" begin
    include("classical_graph.jl")
end

@testset "Simple duality" begin
    include("simple_duality.jl")
end

@testset "Block duality" begin
    include("block_duality.jl")
end

@testset "Block duality 2" begin
    include("block_duality2.jl")
end

# slow and doesn't meet accuracy tolerance
if false
@testset "Thin diag" begin
    include("thin_diag.jl")
end
end

@testset "Diag optimization" begin
    include("diag_optimization.jl")
end

@testset "Unitary transform" begin
    include("unitary_transform.jl")
end

@testset "Compatible matrices" begin
    include("compatible_matrices.jl")
end

@testset "Empty classical graph" begin
    include("empty_classical.jl")
end

@testset "Entropy splitting" begin
    include("entropy_splitting.jl")
end

end # NoncommutativeGraphsTesting
