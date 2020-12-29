NoncommitativeGraphs.jl - Non-commutative graphs in Julia
=========================================================

This package provides support for non-commutative graphs, a quantum analogue
of graphs, as defined in Duan, Severini, Winter,
*Zero-error communication via quantum channels, non-commutative graphs and a quantum
Lovasz theta function*, [arXiv:1002.2514](https://arxiv.org/abs/1002.2514).

Aside from a data structure for holding such graphs, we provide functions for
computing the weighted Lovasz ``\tilde{\vartheta}`` function as defined in
Stahlke, *Weighted theta functions for non-commutative graphs*,
[arXiv:XXX.XXX](https://arxiv.org/abs/XXX.XXX).

FIXME update arxiv reference

## Example

```jldoctest; filter = r"133\.5780.*"
julia> using NoncommutativeGraphs, Random

julia> Random.seed!(0);

julia> sig = [3 2; 2 3]; # S₀ algebra is M₃⊗I₂ ⊕ M₂⊗I₃

julia> S = random_S0Graph(sig)
S0Graph{S0=[3 2; 2 3] S=Subspace{Complex{Float64}} size (12, 12) dim 83}

julia> S.S0 # vertex C*-algebra
Subspace{Complex{Float64}} size (12, 12) dim 13

julia> T = complement(S) # T = perp(S) + S₀
S0Graph{S0=[3 2; 2 3] S=Subspace{Complex{Float64}} size (12, 12) dim 74}

julia> W = randn(ComplexF64, S.n, S.n); W = W' * W; # random weight operator

julia> opt1 = dsw(S, W, eps=1e-7).λ # compute weighted theta
133.57806623525727

julia> opt2 = dsw_via_complement(complement(S), W, eps=1e-7).λ # compute weighted θ via the complement graph, using theorem 29 of arxiv:XXX.XXX.
133.57806730600717

julia> abs(opt1 - opt2) / abs(opt1 + opt2) < 1e-6
true
```
