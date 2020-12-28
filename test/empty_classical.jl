Random.seed!(0)

n = 4
diags = Subspace([ Array{ComplexF64, 2}(basis_vec((n,n), (i,i))) for i in 1:n ])
S = S0Graph([1 n], diags)

w = random_bounded(n)
λ = dsw(S, w).λ

X = HermitianSemidefinite(n)
problem = maximize(real(tr(X * w)), [ X[i,i] == 1 for i in 1:n ])
solve!(problem, () -> SCS.Optimizer(verbose=0, eps=solver_eps))
@test λ ≈ problem.optval  atol=tol
