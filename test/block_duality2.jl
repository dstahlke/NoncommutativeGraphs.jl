Random.seed!(0)

#sig = [1 2; 2 2]
#sig = [1 1; 2 3]
#sig = [2 3]
#sig = [2 2]
sig = [3 2; 2 3]

S = random_S0Graph(sig)
T = complement(S)

w = random_element(S.S1)
w = w*w'
@test w in S.S1

@time opt0 = dsw(S, w, eps=eps)[1]
@time opt1, x, y = dsw_via_complement(T, w, eps=eps)
@test opt1 ≈ opt0  atol=tol
@time opt2 = dsw(T, y, eps=eps)[1]
@test opt2 ≈ 1  atol=tol
@test real(tr(w * √S.D * y * √S.D)) ≈ opt0 * opt2  atol=tol
