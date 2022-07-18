module NoncommutativeGraphs

import Base.==

using DocStringExtensions
using Subspaces
using Convex, SCS, LinearAlgebra
using Random, RandomMatrices
using Graphs
using Compat
using MathOptInterface
import Base.show

export AlgebraShape
export S0Graph
export create_S0_S1
export random_S0Graph, empty_S0Graph, complement, vertex_graph, forget_S0
export from_block_spaces, get_block_spaces
export block_expander
export random_S0_unitary, random_S0_density
export random_S1_unitary, random_S1_density

export Ψ
export dsw_schur, dsw_schur2
export dsw, dsw_via_complement

MOI = MathOptInterface

eye(n) = Matrix(1.0*I, (n,n))

function make_optimizer(verbose, eps)
    optimizer = SCS.Optimizer()
    if isdefined(MOI, :RawOptimizerAttribute) # as of MathOptInterface v0.10.0
        MOI.set(optimizer, MOI.RawOptimizerAttribute("verbose"), verbose)
        MOI.set(optimizer, MOI.RawOptimizerAttribute("eps_rel"), eps)
        MOI.set(optimizer, MOI.RawOptimizerAttribute("eps_abs"), eps)
    else
        MOI.set(optimizer, MOI.RawParameter("verbose"), verbose)
        MOI.set(optimizer, MOI.RawParameter("eps_rel"), eps)
        MOI.set(optimizer, MOI.RawParameter("eps_abs"), eps)
    end
    return optimizer
end

"""
The structure of a finite dimensional C*-algebra.

- For example, `[1 2; 3 4]` corresponds to S₀ = M₁⊗I₂ ⊕ M₃⊗I₄.
- For an n-dimensional non-commutative graph use `[1 n]` for S₀ = Iₙ.
- For an n-vertex classical graph use `ones(Integer, n, 2)` for S₀ = diagonals.
"""
AlgebraShape = Array{<:Integer, 2}

"""
    create_S0_S1(sig::AlgebraShape) -> Tuple{Subspace, Subspace}

Create a C*-algebra and its commutant with the given structure.
"""
function create_S0_S1(sig::AlgebraShape)
    blocks0 = []
    blocks1 = []
    @compat for row in eachrow(sig)
        if length(row) != 2
            throw(ArgumentError("row length must be 2"))
        end
        dA = row[1]
        dY = row[2]
        #println("dA=$dA dY=$dY")
        blk0 = kron(full_subspace(ComplexF64, (dA, dA)), Matrix((1.0+0.0im)*I, dY, dY))
        blk1 = kron(Matrix((1.0+0.0im)*I, dA, dA), full_subspace(ComplexF64, (dY, dY)))
        #println(blk0)
        push!(blocks0, blk0)
        push!(blocks1, blk1)
    end

    S0 = cat(blocks0..., dims=(1,2))
    S1 = cat(blocks1..., dims=(1,2))

    @assert I in S0
    @assert I in S1
    s0 = random_element(S0)
    s1 = random_element(S1)
    @assert (norm(s0 * s1 - s1 * s0) < 1e-9) "S0 and S1 don't commute"

    return S0, S1
end

"""
    S0Graph(sig::AlgebraShape, S::Subspace{ComplexF64, 2})

    S0Graph(g::AbstractGraph)

Represents an S₀-graph as defined in arxiv:1002.2514.

$(TYPEDFIELDS)
"""
struct S0Graph
    """Dimension of Hilbert space A such that S ⊆ L(A)"""
    n::Integer
    """Structure of C*-algebra S₀"""
    sig::AlgebraShape
    """Subspace that defines the graph"""
    S::Subspace{ComplexF64, 2}
    """C*-algebra S₀"""
    S0::Subspace{ComplexF64, 2}
    """Commutant of C*-algebra S₀"""
    S1::Subspace{ComplexF64, 2}
    """Block scaling array D from definition 23 of arxiv:2101.00162"""
    D::Array{Float64, 2}

    function S0Graph(sig::AlgebraShape, S::Subspace{ComplexF64, 2})
        S0, S1 = create_S0_S1(sig)
        S == S' || throw(DomainError("S is not an S0-graph"))
        S0 in S || throw(DomainError("S is not an S0-graph"))
        (S == S0 * S * S0) || throw(DomainError("S is not an S0-graph"))

        n = size(S0)[1]
        da_sizes = sig[:,1]
        dy_sizes = sig[:,2]
        n_sizes = da_sizes .* dy_sizes
        D = cat([ v*eye(n) for (n, v) in zip(n_sizes, dy_sizes ./ da_sizes) ]..., dims=(1,2))

        return new(n, sig, S, S0, S1, D)
    end

    function S0Graph(g::AbstractGraph)
        n = nv(g)
        S0 = Subspace([ (x=zeros(ComplexF64, n,n); x[i,i]=1; x) for i in 1:n ])
        S  = Subspace([ (x=zeros(ComplexF64, n,n); x[src(e),dst(e)]=1; x) for e in edges(g) ])
        S = S + S' + S0
        return S0Graph(ones(Int64, n, 2), S)
    end
end

function show(io::IO, g::S0Graph)
    print(io, "S0Graph{S0=$(g.sig) S=$(g.S)}")
end

"""
$(TYPEDSIGNATURES)

Returns the S₀-graph with S=S₀.
"""
vertex_graph(g::S0Graph) = S0Graph(g.sig, g.S0)

"""
$(TYPEDSIGNATURES)

Returns an S₀-graph with S₀=ℂI.
"""
forget_S0(g::S0Graph) = S0Graph([1 g.n], g.S)

"""
    random_S0Graph(sig::AlgebraShape) -> S0Graph

Creates a random S₀-graph with S₀ having the given structure.
"""
function random_S0Graph(sig::AlgebraShape)
    S0, S1 = create_S0_S1(sig)

    num_blocks = size(sig, 1)
    function block(col, row)
        da_col, dy_col = sig[col,:]
        da_row, dy_row = sig[row,:]
        ds = Integer(round(dy_row * dy_col / 2.0))
        F = full_subspace(ComplexF64, (da_row, da_col))
        if row == col
            R = random_hermitian_subspace(ComplexF64, ds, dy_row)
        elseif row > col
            R = random_subspace(ComplexF64, ds, (dy_row, dy_col))
        else
            R = empty_subspace(ComplexF64, (dy_row, dy_col))
        end
        return kron(F, R)
    end
    blocks = Array{Subspace{ComplexF64, 2}, 2}([
        block(col, row)
        for col in 1:num_blocks, row in 1:num_blocks
   ])

    S = hvcat(num_blocks, blocks...)
    S |= S'
    S |= S0

    return S0Graph(sig, S)
end

"""
    empty_S0Graph(sig::AlgebraShape) -> S0Graph

Creates an empty S₀-graph (i.e. S=S₀) with S₀ having the given structure.
"""
function empty_S0Graph(sig::AlgebraShape)
    S0, S1 = create_S0_S1(sig)
    return S0Graph(sig, S0)
end

"""
$(TYPEDSIGNATURES)

Returns the complement graph perp(S) + S₀.
"""
complement(g::S0Graph) = S0Graph(g.sig, perp(g.S) | g.S0)

function ==(a::S0Graph, b::S0Graph)
    return a.sig == b.sig && a.S == b.S
end

function get_block_spaces(g::S0Graph)
    num_blocks = size(g.sig, 1)
    da_sizes = g.sig[:,1]
    dy_sizes = g.sig[:,2]
    n_sizes = da_sizes .* dy_sizes

    blkspaces = Array{Subspace{ComplexF64}, 2}(undef, num_blocks, num_blocks)
    offseti = 0
    for blki in 1:num_blocks
        offsetj = 0
        for blkj in 1:num_blocks
            #@show [blki, blkj, offseti, offsetj]
            blkbasis = Array{Array{ComplexF64, 2}, 1}()
            for m in each_basis_element(g.S)
                blk = m[1+offseti:dy_sizes[blki]+offseti, 1+offsetj:dy_sizes[blkj]+offsetj]
                push!(blkbasis, blk)
            end
            blkspaces[blki, blkj] = Subspace(blkbasis)
            #println(blkspaces[blki, blkj])
            offsetj += n_sizes[blkj]
        end
        @assert offsetj == size(g.S)[2]
        offseti += n_sizes[blki]
    end
    @assert offseti == size(g.S)[1]

    return blkspaces
end

function from_block_spaces(sig::AlgebraShape, blkspaces::Array{Subspace{ComplexF64}, 2})
    S0, S1 = NoncommutativeGraphs.create_S0_S1(sig)

    num_blocks = size(sig, 1)
    function block(col, row)
        da_col, dy_col = sig[col,:]
        da_row, dy_row = sig[row,:]
        ds = Integer(round(sqrt(dy_row * dy_col) / 2.0))
        F = full_subspace(ComplexF64, (da_row, da_col))
        kron(F, blkspaces[row, col])
    end
    blocks = [
        block(col, row)
        for col in 1:num_blocks, row in 1:num_blocks
    ]

    S = hvcat(num_blocks, blocks...)
    S |= S'
    S |= S0

    return S0Graph(sig, S)
end

function block_expander(sig::AlgebraShape)
    function basis_mat(n, i)
        M = zeros(n*n)
        M[i] = 1
        return reshape(M, (n, n))
    end

    n = sum(prod(sig, dims=2))

    @compat J = cat([
        cat([ kron(eye(dA), basis_mat(dY, i)) for i in 1:dY^2 ]..., dims=3)
        for (dA, dY) in eachrow(sig)
    ]..., dims=(1,2,3))

    return reshape(J, (n^2, size(J)[3]))
end

function random_positive(n)
    U = rand(Haar(2), n)
    return Hermitian(U' * Diagonal(rand(n)) * U)
end

"""
Returns a random unitary in S₀.
"""
function random_S0_unitary(sig::AlgebraShape)
    @compat return cat([
        kron(rand(Haar(2), dA), eye(dY))
        for (dA, dY) in eachrow(sig)
    ]..., dims=(1,2))
end

"""
Returns a random density operator in S₀.
"""
function random_S0_density(sig::AlgebraShape)
    @compat ρ = cat([
        kron(random_positive(dA), eye(dY))
        for (dA, dY) in eachrow(sig)
    ]..., dims=(1,2))
    return ρ / tr(ρ)
end

"""
Returns a random unitary in the commutant of S₀.
"""
function random_S1_unitary(sig::AlgebraShape)
    @compat return cat([
        kron(eye(dA), rand(Haar(2), dY))
        for (dA, dY) in eachrow(sig)
    ]..., dims=(1,2))
end

"""
Returns a random density operator in the commutant of S₀.
"""
function random_S1_density(sig::AlgebraShape)
    @compat ρ = cat([
        kron(eye(dA), random_positive(dY))
        for (dA, dY) in eachrow(sig)
    ]..., dims=(1,2))
    return ρ / tr(ρ)
end

###############
### DSW solvers
###############

# FIXME Convex.jl doesn't support cat(args..., dims=(1,2)).  It should be added.
function diagcat(args::Convex.AbstractExprOrValue...)
    num_blocks = size(args, 1)
    return vcat([
        hcat([
            row == col ? args[row] : zeros(size(args[row], 1), size(args[col], 2))
            for col in 1:num_blocks
        ]...)
        for row in 1:num_blocks
    ]...)
end

"""
$(TYPEDSIGNATURES)

Block scaling superoperator Ψ from definition 23 of arxiv:2101.00162
"""
function Ψ(g::S0Graph, w::Union{AbstractArray{<:Number, 2}, Variable})
    n = g.n
    da_sizes = g.sig[:,1]
    dy_sizes = g.sig[:,2]
    n_sizes = da_sizes .* dy_sizes
    num_blocks = size(g.sig)[1]

    k = 0
    blocks = []
    for (dai, dyi) in zip(da_sizes, dy_sizes)
        ni = dai * dyi
        TrAi = partialtrace(w[k+1:k+ni, k+1:k+ni], 1, [dai; dyi])
        blk = dyi^-1 * kron(Array(1.0*I, dai, dai), TrAi)
        k += ni
        push!(blocks, blk)
    end
    @assert k == n
    out = diagcat(blocks...)
    @assert size(out) == (n, n)
    return out
end

"""
$(TYPEDSIGNATURES)

Schur complement form of weighted θ from theorem 14 of arxiv:2101.00162.

Returns λ, w, and Z variables (for Convex.jl) in a named tuple.

See also: [`dsw_schur2`](@ref).
"""
function dsw_schur(g::S0Graph)::NamedTuple{(:λ, :w, :Z), Tuple{Convex.AbstractExpr, Convex.AbstractExpr, Convex.AbstractExpr}}
    n = size(g.S)[1]

    Z = sum(kron(m, ComplexVariable(n, n)) for m in hermitian_basis(g.S))
    # slow:
    #Z = ComplexVariable(n^2, n^2)
    #add_constraint!(Z, Z in kron(g.S, full_subspace(ComplexF64, (n, n))))
    # slow:
    #SB = kron(g.S, full_subspace(ComplexF64, (n, n)))
    #Z = variable_in_space(SB)

    λ = Variable()
    wt = partialtrace(Z, 1, [n; n])
    wv = reshape(wt, n*n, 1)

    add_constraint!(λ, [ λ  wv' ; wv  Z ] ⪰ 0)

    return (λ=λ, w=transpose(wt), Z=Z)
end

"""
$(TYPEDSIGNATURES)

Schur complement form of weighted θ from theorem 14 of arxiv:2101.00162, optimized for the
case S₀ ≠ ℂI, at the cost of w being constrained to S₁ (the commutant of S₀).

Returns λ, w, and Z variables (for Convex.jl) in a named tuple.

See also: [`dsw_schur2`](@ref).
"""
function dsw_schur2(g::S0Graph)::NamedTuple{(:λ, :w, :Z), Tuple{Convex.AbstractExpr, Convex.AbstractExpr, Convex.AbstractExpr}}
    da_sizes = g.sig[:,1]
    dy_sizes = g.sig[:,2]
    num_blocks = size(g.sig)[1]

    blkspaces = get_block_spaces(g)

    w_blocks = Array{Convex.AbstractExpr, 1}(undef, num_blocks)
    Z_blocks = Array{Convex.AbstractExpr, 2}(undef, num_blocks, num_blocks)
    for blki in 1:num_blocks
        for blkj in 1:num_blocks
            if blkj <= blki
                if dim(blkspaces[blki, blkj]) == 0
                    Z_blocks[blki, blkj] = zeros(dy_sizes[blki]^2, dy_sizes[blkj]^2)
                else
                    blkV = sum(kron(m, ComplexVariable(dy_sizes[blki], dy_sizes[blkj]))
                        for m in each_basis_element(blkspaces[blki, blkj]))
                    Z_blocks[blki, blkj] = blkV
                    # slow:
                    #SB = kron(blkspaces[blki, blkj], full_subspace(ComplexF64, (dy_sizes[blki], dy_sizes[blkj])))
                    #Z_blocks[blki, blkj] = variable_in_space(SB)
                end
            end
            if blkj == blki
                w_blocks[blki] = partialtrace(Z_blocks[blki, blkj], 1, [dy_sizes[blki], dy_sizes[blkj]])
            end
        end
    end

    for blki in 1:num_blocks
        for blkj in 1:num_blocks
            if blkj > blki
                Z_blocks[blki, blkj] = Z_blocks[blkj, blki]'
            end
            #@show blki, blkj
            #@show size(Z_blocks[blki, blkj])
        end
    end

    #@show [ size(wi) for wi in w_blocks ]

    λ = Variable()
    Z = vcat([ hcat([Z_blocks[i,j] for j in 1:num_blocks]...) for i in 1:num_blocks ]...)
    wv = vcat([ reshape(wi, dy_sizes[i]^2, 1) for (i,wi) in enumerate(w_blocks) ]...)
    #@show size(wv)
    #@show size(Z)

    add_constraint!(λ, [ λ  wv' ; wv  Z ] ⪰ 0)

    wt = diagcat([ kron(eye(da_sizes[i]), wi) for (i,wi) in enumerate(w_blocks) ]...)
    #@show size(wt)

    return (λ=λ, w=transpose(wt), Z=Z)
end

"""
$(TYPEDSIGNATURES)

Compute weighted θ using theorem 14 of arxiv:2101.00162.

Returns optimal λ, x, and Z values in a named tuple.
If `use_diag_optimization=true` (the default) then `x ⪰ w` and `x` is in the commutant
of S₀.  By theorem 29 of arxiv:2101.00162, θ(g, w) = θ(g, x).
"""
function dsw(g::S0Graph, w::AbstractArray{<:Number, 2}; use_diag_optimization=true, eps=1e-6, verbose=0)
    if use_diag_optimization
        λ, x, Z = dsw_schur2(g)
    else
        λ, x, Z = dsw_schur(g)
    end

    problem = minimize(λ, [x ⪰ w])
    solve!(problem, () -> make_optimizer(verbose, eps))
    return (λ=problem.optval, x=Hermitian(evaluate(x)), Z=Hermitian(evaluate(Z)))
end

"""
$(TYPEDSIGNATURES)

Compute weighted θ via the complement graph, using theorem 29 of arxiv:2101.00162.

θ(S, w) = max{ tr(w x) : x ⪰ 0, y = Ψ(x), θ(Sᶜ, y) ≤ 1 }

Returns optimal λ, x, y, and Z in a named tuple.

If w is in the commutant of S₀ then the weights w and y saturate the inequality in
theorem 32 of arxiv:2101.00162.
"""
function dsw_via_complement(g::S0Graph, w::AbstractArray{<:Number, 2}; use_diag_optimization=true, eps=1e-6, verbose=0)
    # max{ <w,x> : Ψ(S, x) ⪯ y, ϑ(S, y) ≤ 1, y ∈ S1 }
    # equal to:
    # max{ dsw(S0, √y * w * √y) : dsw(complement(S), y) <= 1 }
    if use_diag_optimization
        λ, y, Z = dsw_schur2(g)
    else
        λ, y, Z = dsw_schur(g)
    end
    x = HermitianSemidefinite(g.n, g.n)
    problem = maximize(real(tr(w * x')), [ λ <= 1, Ψ(g, x) == y ])
    solve!(problem, () -> make_optimizer(verbose, eps))
    return (λ=problem.optval, x=Hermitian(evaluate(x)), y=Hermitian(evaluate(y)), Z=Hermitian(evaluate(Z)))
end

end
