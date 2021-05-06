sig = [2 3; 3 2; 1 1]
S = random_S0Graph(sig)

w = random_S0_density(S.sig)
w /= tr(w)

#@show S
#display(w)

λ, x1, Z = dsw_schur2(S)
problem = maximize(trace_logm(x1, w), [ λ <= 1 ])
@time solve!(problem, () -> SCS.Optimizer(verbose=0, eps=solver_eps))
h1=problem.optval
x1=Hermitian(evaluate(x1))

λ, x2, Z = dsw_schur2(complement(S))
problem = maximize(trace_logm(x2, w), [ λ <= 1 ])
@time solve!(problem, () -> SCS.Optimizer(verbose=0, eps=solver_eps))
h2=problem.optval
x2=Hermitian(evaluate(x2))

@test x1*x2 ≈ x2*x1  atol=tol
@test x1*x2 ≈ Ψ(S, w)  atol=tol
# follows from the above
@test h1+h2 ≈ tr(w*log(Ψ(S, w)))  atol=tol
