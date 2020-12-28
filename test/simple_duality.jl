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
@time opt2 = dsw_via_complement(complement(S), w, eps=eps)[1]
@test opt1 ≈ opt2  atol=tol
