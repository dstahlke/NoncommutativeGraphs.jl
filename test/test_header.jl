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

solver_eps = 1e-7
tol = 1e-6
make_optimizer = NoncommutativeGraphs.make_optimizer
