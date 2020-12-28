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
