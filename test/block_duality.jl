function f(g, w, v)
    n = size(w, 1)
    q = HermitianSemidefinite(n)
    problem = maximize(real(tr(w * q')), [ Ψ(g, q) ⪯ v ])
    solve!(problem, () -> make_optimizer(0, solver_eps))
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

    solve!(problem, () -> make_optimizer(0, solver_eps))
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
#    solve!(problem, () -> make_optimizer(0, solver_eps))
#    return problem.optval
#end

Random.seed!(0)

#sig = [1 2; 2 2]
#sig = [1 1; 2 3]
#sig = [2 3]
#sig = [2 2]
sig = [3 2; 2 3]

S = random_S0Graph(sig)
T = complement(S)

w = random_bounded(S.n)

@time opt0 = dsw(S, w, eps=solver_eps)[1]
@time opt1, x, y = dsw_via_complement(T, w, eps=solver_eps)
@test opt1 ≈ opt0  atol=tol
@time opt2 = dsw(T, y, eps=solver_eps)[1]
@test opt2 ≈ 1  atol=tol
@time opt3 = dsw(vertex_graph(S), √y * w * √y, eps=solver_eps)[1]
@test opt3 ≈ opt0  atol=tol
# ϑ(S, w) = max{ ϑ(S0, √y * w * √y) / ϑ(T, y) : y ∈ S1 }
@test opt0 ≈ opt3 / opt2  atol=tol

@test f(S, w, y) ≈ opt3  atol=tol
@test g(S, √y * w * √y) ≈ opt3  atol=tol
