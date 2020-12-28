Random.seed!(0)

#sig = [1 2; 2 2]
#sig = [1 1; 2 3]
sig = [2 3]
#sig = [2 2]
#sig = [3 2; 2 3]

S = random_S0Graph(sig)
Sthin = complement(forget_S0(complement(S)))

w = random_bounded(S.n)

@time opt1 = dsw(Sthin, w, eps=solver_eps, verbose=1)[1]
@time opt2 = dsw(S, Ψ(S, w), eps=solver_eps)[1]
@test opt1 ≈ opt2*S.n  atol=tol
