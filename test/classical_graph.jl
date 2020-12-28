using LightGraphs.cycle_graph

n = 7
G = cycle_graph(n)
S = S0Graph(G)
@time λ = dsw(S, eye(n), eps=eps)[1]
@test λ ≈ n*cos(pi/n) / (1 + cos(pi/n))  atol=tol
