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
@time opt1 = dsw(S, w, eps=solver_eps)[1]
@time opt2 = dsw(SU, U*w*U', eps=solver_eps)[1]
@test opt1 ≈ opt2  atol=tol
