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
    @time opt1, _, x2, Z2 = dsw_via_complement(T, w, eps=eps)
    @test opt1 ≈ opt0  atol=tol
else
    @time opt1, _, y = dsw_via_complement(T, w, eps=eps)
    @test opt1 ≈ opt0  atol=tol
    @time opt2, x2, Z2 = dsw(T, y, eps=eps)
    @test opt2 ≈ 1  atol=tol
end

x1 /= opt0
Z1 /= opt0
Z1 = J * Z1 * J'
Z2 = J * Z2 * J'
@test Z1 ≈ Z1'
@test Z2 ≈ Z2'
Z1 = Hermitian(Z1)
Z2 = Hermitian(Z2)

@test tr(√D * x1 * √D * x2) ≈ 1  atol=tol
@test partialtrace(Z1, 1, [n,n]) ≈ transpose(x1)  atol=tol
@test partialtrace(Z2, 1, [n,n]) ≈ transpose(x2)  atol=tol

v1 = reshape(conj(x1), n^2)
v2 = reshape(conj(x2), n^2)
Q1 = [1 v1'; v1 Z1]
Q2 = [1 v2'; v2 Z2]
@test Q1 ≈ Q1'
@test Q2 ≈ Q2'
Q1 = Hermitian(Q1)
Q2 = Hermitian(Q2)
D2 = cat([ 1, -kron(√D, √D) ]..., dims=(1,2))

@test minimum(eigvals(Q1)) > -tol
@test minimum(eigvals(Q2)) > -tol
@test tr(Q1 * D2 * Q2 * D2) ≈ 0  atol=tol
@test Z1 * kron(√D, √D) * v2 ≈ v1  atol=tol
@test Z2 * kron(√D, √D) * v1 ≈ v2  atol=tol
@test Z1 * kron(√D, √D) * Z2 ≈ v1 * v2'  atol=tol
@test Z1 * kron(eye(n), D) * Z2 ≈ v1 * v2'  atol=tol
