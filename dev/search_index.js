var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [NoncommutativeGraphs]\nPrivate = false","category":"page"},{"location":"reference/#NoncommutativeGraphs.AlgebraShape","page":"Reference","title":"NoncommutativeGraphs.AlgebraShape","text":"The structure of a finite dimensional C*-algebra.\n\nFor example, [1 2; 3 4] corresponds to S₀ = M₁⊗I₂ ⊕ M₃⊗I₄.\nFor an n-dimensional non-commutative graph use [1 n] for S₀ = Iₙ.\nFor an n-vertex classical graph use ones(Integer, n, 2) for S₀ = diagonals.\n\n\n\n\n\n","category":"type"},{"location":"reference/#NoncommutativeGraphs.S0Graph","page":"Reference","title":"NoncommutativeGraphs.S0Graph","text":"S0Graph(sig::AlgebraShape, S::Subspace{ComplexF64, 2})\n\nS0Graph(g::AbstractGraph)\n\nRepresents an S₀-graph as defined in arxiv:1002.2514.\n\nn::Integer\nDimension of Hilbert space A such that S ⊆ L(A)\nsig::Array{var\"#s12\",2} where var\"#s12\"<:Integer\nStructure of C*-algebra S₀\nS::Subspaces.Subspace{Complex{Float64},2}\nSubspace that defines the graph\nS0::Subspaces.Subspace{Complex{Float64},2}\nC*-algebra S₀\nS1::Subspaces.Subspace{Complex{Float64},2}\nCommutant of C*-algebra S₀\nD::Array{Float64,2}\nBlock scaling array D from definition 23 of arxiv:2101.00162\n\n\n\n\n\n","category":"type"},{"location":"reference/#NoncommutativeGraphs.complement-Tuple{S0Graph}","page":"Reference","title":"NoncommutativeGraphs.complement","text":"complement(g::S0Graph) -> S0Graph\n\n\nReturns the complement graph perp(S) + S₀.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.create_S0_S1-Tuple{Array{var\"#s12\",2} where var\"#s12\"<:Integer}","page":"Reference","title":"NoncommutativeGraphs.create_S0_S1","text":"create_S0_S1(sig::AlgebraShape) -> Tuple{Subspace, Subspace}\n\nCreate a C*-algebra and its commutant with the given structure.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.dsw-Tuple{S0Graph,AbstractArray{var\"#s13\",2} where var\"#s13\"<:Number}","page":"Reference","title":"NoncommutativeGraphs.dsw","text":"dsw(g::S0Graph, w::AbstractArray{var\"#s13\",2} where var\"#s13\"<:Number; use_diag_optimization, eps, verbose) -> NamedTuple{(:λ, :x, :Z),_A} where _A<:Tuple\n\n\nCompute weighted θ using theorem 14 of arxiv:2101.00162.\n\nReturns optimal λ, x, and Z values in a named tuple. If use_diag_optimization=true (the default) then x ⪰ w and x is in the commutant of S₀.  By theorem 29 of arxiv:2101.00162, θ(g, w) = θ(g, x).\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.dsw_schur-Tuple{S0Graph}","page":"Reference","title":"NoncommutativeGraphs.dsw_schur","text":"dsw_schur(g::S0Graph) -> NamedTuple{(:λ, :w, :Z),Tuple{Convex.AbstractExpr,Convex.AbstractExpr,Convex.AbstractExpr}}\n\n\nSchur complement form of weighted θ from theorem 14 of arxiv:2101.00162.\n\nReturns λ, w, and Z variables (for Convex.jl) in a named tuple.\n\nSee also: dsw_schur2.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.dsw_schur2-Tuple{S0Graph}","page":"Reference","title":"NoncommutativeGraphs.dsw_schur2","text":"dsw_schur2(g::S0Graph) -> NamedTuple{(:λ, :w, :Z),Tuple{Convex.AbstractExpr,Convex.AbstractExpr,Convex.AbstractExpr}}\n\n\nSchur complement form of weighted θ from theorem 14 of arxiv:2101.00162, optimized for the case S₀ ≠ ℂI, at the cost of w being constrained to S₁ (the commutant of S₀).\n\nReturns λ, w, and Z variables (for Convex.jl) in a named tuple.\n\nSee also: dsw_schur2.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.dsw_via_complement-Tuple{S0Graph,AbstractArray{var\"#s17\",2} where var\"#s17\"<:Number}","page":"Reference","title":"NoncommutativeGraphs.dsw_via_complement","text":"dsw_via_complement(g::S0Graph, w::AbstractArray{var\"#s17\",2} where var\"#s17\"<:Number; use_diag_optimization, eps, verbose) -> NamedTuple{(:λ, :x, :y, :Z),_A} where _A<:Tuple\n\n\nCompute weighted θ via the complement graph, using theorem 29 of arxiv:2101.00162.\n\nθ(S, w) = max{ tr(w x) : x ⪰ 0, y = Ψ(x), θ(Sᶜ, y) ≤ 1 }\n\nReturns optimal λ, x, y, and Z in a named tuple.\n\nIf w is in the commutant of S₀ then the weights w and y saturate the inequality in theorem 32 of arxiv:2101.00162.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.empty_S0Graph-Tuple{Array{var\"#s12\",2} where var\"#s12\"<:Integer}","page":"Reference","title":"NoncommutativeGraphs.empty_S0Graph","text":"empty_S0Graph(sig::AlgebraShape) -> S0Graph\n\nCreates an empty S₀-graph (i.e. S=S₀) with S₀ having the given structure.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.forget_S0-Tuple{S0Graph}","page":"Reference","title":"NoncommutativeGraphs.forget_S0","text":"forget_S0(g::S0Graph) -> S0Graph\n\n\nReturns an S₀-graph with S₀=ℂI.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.random_S0Graph-Tuple{Array{var\"#s12\",2} where var\"#s12\"<:Integer}","page":"Reference","title":"NoncommutativeGraphs.random_S0Graph","text":"random_S0Graph(sig::AlgebraShape) -> S0Graph\n\nCreates a random S₀-graph with S₀ having the given structure.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.random_S1_unitary-Tuple{Array{var\"#s12\",2} where var\"#s12\"<:Integer}","page":"Reference","title":"NoncommutativeGraphs.random_S1_unitary","text":"Returns a random unitary in the commutant of S₀.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.vertex_graph-Tuple{S0Graph}","page":"Reference","title":"NoncommutativeGraphs.vertex_graph","text":"vertex_graph(g::S0Graph) -> S0Graph\n\n\nReturns the S₀-graph with S=S₀.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NoncommutativeGraphs.Ψ-Tuple{S0Graph,Union{Convex.Variable, AbstractArray{var\"#s13\",2} where var\"#s13\"<:Number}}","page":"Reference","title":"NoncommutativeGraphs.Ψ","text":"Ψ(g::S0Graph, w::Union{Convex.Variable, AbstractArray{var\"#s13\",2} where var\"#s13\"<:Number}) -> Any\n\n\nBlock scaling superoperator Ψ from definition 23 of arxiv:2101.00162\n\n\n\n\n\n","category":"method"},{"location":"#NoncommitativeGraphs.jl-Non-commutative-graphs-in-Julia","page":"Home","title":"NoncommitativeGraphs.jl - Non-commutative graphs in Julia","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides support for non-commutative graphs, a quantum analogue of graphs, as defined in Duan, Severini, Winter, Zero-error communication via quantum channels, non-commutative graphs and a quantum Lovasz theta function, arXiv:1002.2514.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Aside from a data structure for holding such graphs, we provide functions for computing the weighted Lovasz theta function as defined in Stahlke, Weighted theta functions for non-commutative graphs, arXiv:2101.00162.","category":"page"},{"location":"","page":"Home","title":"Home","text":"FIXME update arxiv reference","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> using NoncommutativeGraphs, Random\n\njulia> Random.seed!(0);\n\njulia> sig = [3 2; 2 3]; # S₀ algebra is M₃⊗I₂ ⊕ M₂⊗I₃\n\njulia> S = random_S0Graph(sig)\nS0Graph{S0=[3 2; 2 3] S=Subspace{Complex{Float64}} size (12, 12) dim 83}\n\njulia> S.S0 # vertex C*-algebra\nSubspace{Complex{Float64}} size (12, 12) dim 13\n\njulia> T = complement(S) # T = perp(S) + S₀\nS0Graph{S0=[3 2; 2 3] S=Subspace{Complex{Float64}} size (12, 12) dim 74}\n\njulia> W = randn(ComplexF64, S.n, S.n); W = W' * W; # random weight operator\n\njulia> opt1 = dsw(S, W, eps=1e-7).λ # compute weighted theta\n133.57806623525727\n\njulia> opt2 = dsw_via_complement(complement(S), W, eps=1e-7).λ # compute weighted θ via the complement graph, using theorem 29 of arxiv:2101.00162.\n133.57806730600717\n\njulia> abs(opt1 - opt2) / abs(opt1 + opt2) < 1e-6\ntrue","category":"page"}]
}
