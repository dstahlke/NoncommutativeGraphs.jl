NoncommitativeGraphs.jl - Non-commutative graphs in Julia
=========================================================

FIXME update this from docs/src/index.md

This package provides support for non-commutative graphs, a quantum analogue
of graphs, as defined in Duan, Severini, Winter,
*Zero-error communication via quantum channels, non-commutative graphs and a quantum
Lovasz theta function*, [arXiv:1002.2514](https://arxiv.org/abs/1002.2514).

Aside from a data structure for holding such graphs, we provide functions for
computing the weighted Lovasz \\( \tilde{\vartheta} \\) function.

## Example

```julia
julia> using NoncommutativeGraphs
julia> sig = [3 2; 2 3] # S_0 algebra is \\( M_3 \otimes I_2 \oplus M_2 \otimes I_3 \\)

julia> S = random_S0Graph(sig)
julia> T = complement(S) # \\( T = S^\perp + S_0 \\)

julia> w = random_bounded(S.n) # weight vector

julia> opt1 = dsw(S, w)[1]
julia> opt2 = dsw_antiblocker(complement(S), w)[1]
julia> opt1 ≈ opt2  atol=tol
```
