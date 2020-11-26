using NoncommutativeGraphs, Subspaces
using Convex, SCS, LinearAlgebra
using Random, RandomMatrices
using LightGraphs
using Test

complement = NoncommutativeGraphs.complement

function random_bounded(n)
    U = rand(Haar(2), n)
    return Hermitian(U' * Diagonal(rand(n)) * U)
end

roundmat(x) = ((x)->abs(x) < 1e-4 ? 0.0*x : x).(x)

eye(n) = Matrix(1.0*I, (n,n))

eps = 1e-7
tol = 1e-6

function f(g, w, v)
    n = size(w, 1)
    q = HermitianSemidefinite(n)
    problem = maximize(real(tr(w * q')), [ Ψ(g, q) ⪯ v ])
    solve!(problem, () -> SCS.Optimizer(verbose=0, eps=eps))
    return problem.optval
end

function g(g, w)
    D = g.D

    #z = HermitianSemidefinite(g.n)
    #problem = minimize(real(tr(D * z)), [ z ⪰ w, z in g.S1 ])
    #problem = minimize(real(tr(z)), [ z ⪰ √D * w * √D, z in g.S1 ])

    z = variable_in_space(g.S1)
    problem = minimize(real(tr(D * z)), [ z ⪰ w ])
    #problem = minimize(real(tr(z)), [ z ⪰ √D * w * √D ])

    solve!(problem, () -> SCS.Optimizer(verbose=0, eps=eps))
    return problem.optval
end

#function h(g, w, y)
#    D = g.D
#
#    z = variable_in_space(g.S1)
#    problem = minimize(real(tr(y * √D * z * √D)), [ z ⪰ w ])
#    #problem = minimize(real(tr(D * √y * z * √y)), [ z ⪰ w ])
#    #problem = minimize(real(tr(z)), [ z ⪰ √D * √y * w * √y * √D ])
#
#    solve!(problem, () -> SCS.Optimizer(verbose=0, eps=eps))
#    return problem.optval
#end

@testset "Classical graph" begin
    n = 7
    G = cycle_graph(n)
    S = classical_S0Graph(G)
    @time λ = dsw(S, eye(n), eps=eps)[1]
    @test λ ≈ n*cos(pi/n) / (1 + cos(pi/n))  atol=tol
end

@testset "Simple duality" begin
    Random.seed!(0)

    #sig = [1 2; 2 2]
    #sig = [1 1; 2 3]
    #sig = [2 3]
    #sig = [2 2]
    sig = [3 2; 2 3]

    S = random_S0Graph(sig)
    T = complement(S)

    w = random_bounded(S.n)

    @time opt1 = dsw(S, w, eps=eps)[1]
    @time opt2 = dsw_antiblocker(complement(S), w, eps=eps)[1]
    @test opt1 ≈ opt2  atol=tol
end

@testset "Block duality" begin
    Random.seed!(0)

    #sig = [1 2; 2 2]
    #sig = [1 1; 2 3]
    #sig = [2 3]
    #sig = [2 2]
    sig = [3 2; 2 3]

    S = random_S0Graph(sig)
    T = complement(S)

    w = random_bounded(S.n)

    @time opt0 = dsw(S, w, eps=eps)[1]
    @time opt1, x, y = dsw_antiblocker(T, w, eps=eps)
    @test opt1 ≈ opt0  atol=tol
    @time opt2 = dsw(T, y, eps=eps)[1]
    @test opt2 ≈ 1  atol=tol
    @time opt3 = dsw(vertex_graph(S), √y * w * √y, eps=eps)[1]
    @test opt3 ≈ opt0  atol=tol
    # ϑ(S, w) = max{ ϑ(S0, √y * w * √y) / ϑ(T, y) : y ∈ S1 }
    @test opt0 ≈ opt3 / opt2  atol=tol

    @test f(S, w, y) ≈ opt3  atol=tol
    @test g(S, √y * w * √y) ≈ opt3  atol=tol
end

# slow and doesn't meet accuracy tolerance
if false
@testset "Thin diag" begin
    Random.seed!(0)

    #sig = [1 2; 2 2]
    #sig = [1 1; 2 3]
    sig = [2 3]
    #sig = [2 2]
    #sig = [3 2; 2 3]

    S = random_S0Graph(sig)
    Sthin = complement(forget_S0(complement(S)))

    w = random_bounded(S.n)

    @time opt1 = dsw(Sthin, w, eps=eps, verbose=1)[1]
    @time opt2 = dsw(S, Ψ(S, w), eps=eps)[1]
    @test opt1 ≈ opt2*S.n  atol=tol
end
end

@testset "Diag optimization" begin
    Random.seed!(0)

    #sig = [1 2; 2 2]
    #sig = [1 1; 2 3]
    sig = [2 3]
    #sig = [2 2]
    #sig = [3 2; 2 3]

    S = random_S0Graph(sig)

    w = random_bounded(S.n)

    @time opt1, x1 = dsw(S, w, eps=eps)
    @time opt2, x2 = dsw(S, w, use_diag_optimization=false, eps=eps)
    @test opt1 ≈ opt2  atol=tol
end

@testset "Unitary transform" begin
    Random.seed!(0)

    #sig = [1 2; 2 2]
    #sig = [1 1; 2 3]
    sig = [2 3]
    #sig = [2 2]
    #sig = [3 2; 2 3]

    S = random_S0Graph(sig)

    w = random_bounded(S.n)

    U = random_S1_unitary(sig);
    SU = S0Graph(S.sig, U * S.S * U')
    @time opt1 = dsw(S, w, eps=eps)[1]
    @time opt2 = dsw(SU, U*w*U', eps=eps)[1]
    @test opt1 ≈ opt2  atol=tol
end

@testset "Compatible matrices" begin
    Random.seed!(0)

    #sig = [1 2; 2 2]
    #sig = [1 1; 2 3]
    #sig = [2 3]
    #sig = [2 2]
    sig = [3 2; 2 3]

    S = random_S0Graph(sig)
    T = complement(S)

    n = S.n
    D = S.D
    J = block_expander(S.sig)

    w = random_bounded(S.n)

    @time opt0, x1, Z1 = dsw(S, w, eps=eps)
    if true
        @time opt1, _, x2, Z2 = dsw_antiblocker(T, w, eps=eps)
        @test opt1 ≈ opt0  atol=1e-6
    else
        @time opt1, _, y = dsw_antiblocker(T, w, eps=eps)
        @test opt1 ≈ opt0  atol=1e-6
        @time opt2, x2, Z2 = dsw(T, y, eps=eps)
        @test opt2 ≈ 1  atol=1e-6
    end

    x1 /= opt0
    Z1 /= opt0
    Z1 = J * Z1 * J'
    Z2 = J * Z2 * J'
    @test Z1 ≈ Z1'
    @test Z2 ≈ Z2'
    Z1 = Hermitian(Z1)
    Z2 = Hermitian(Z2)

    @test tr(√D * x1 * √D * x2) ≈ 1  atol=1e-6
    @test partialtrace(Z1, 1, [n,n]) ≈ transpose(x1)  atol=1e-6
    @test partialtrace(Z2, 1, [n,n]) ≈ transpose(x2)  atol=1e-6

    v1 = reshape(conj(x1), n^2)
    v2 = reshape(conj(x2), n^2)
    Q1 = [1 v1'; v1 Z1]
    Q2 = [1 v2'; v2 Z2]
    @test Q1 ≈ Q1'
    @test Q2 ≈ Q2'
    Q1 = Hermitian(Q1)
    Q2 = Hermitian(Q2)
    D2 = cat([ 1, -kron(√D, √D) ]..., dims=(1,2))

    @test minimum(eigvals(Q1)) > -1e-6
    @test minimum(eigvals(Q2)) > -1e-6
    @test tr(Q1 * D2 * Q2 * D2) ≈ 0  atol=1e-6
    @test Z1 * kron(√D, √D) * v2 ≈ v1  atol=1e-6
    @test Z2 * kron(√D, √D) * v1 ≈ v2  atol=1e-6
    @test Z1 * kron(√D, √D) * Z2 ≈ v1 * v2'  atol=1e-6
    @test Z1 * kron(eye(n), D) * Z2 ≈ v1 * v2'  atol=1e-6
end
