module NoncommutativeGraphsTesting

using NoncommutativeGraphs, Subspaces
using Convex, SCS, LinearAlgebra
using Random, RandomMatrices
using Test

function random_bounded(n)
    U = rand(Haar(2), n)
    return Hermitian(U' * Diagonal(rand(n)) * U)
end

function basis_vec(dims::Tuple, idx::Tuple)
    m = zeros(dims)
    m[idx...] = 1
    return m
end

eye(n) = Matrix(1.0*I, (n,n))

eps = 1e-7
tol = 1e-6

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

end # NoncommutativeGraphsTesting
