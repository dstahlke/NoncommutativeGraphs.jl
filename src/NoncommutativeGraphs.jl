module NoncommutativeGraphs

import Base.==

using Subspaces
using Convex, SCS, LinearAlgebra
using Random, RandomMatrices
using LightGraphs
import Base.show

export AlgebraShape
export S0Graph
export create_S0_S1
export random_S0Graph, empty_S0Graph, complement, vertex_graph, forget_S0
export from_block_spaces, get_block_spaces
export block_expander
export random_S1_unitary

export Ψ
export dsw_schur, dsw_schur2
export dsw, dsw_antiblocker

eye(n) = Matrix(1.0*I, (n,n))

AlgebraShape = Array{<:Integer, 2}

function create_S0_S1(sig::AlgebraShape)
    blocks0 = []
    blocks1 = []
    for row in eachrow(sig)
        if length(row) != 2
            throw(ArgumentError("row length must be 2"))
        end
        dA = row[1]
        dY = row[2]
        #println("dA=$dA dY=$dY")
        blk0 = kron(full_subspace((dA, dA)), Matrix((1.0+0.0im)*I, dY, dY))
        blk1 = kron(Matrix((1.0+0.0im)*I, dA, dA), full_subspace((dY, dY)))
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

struct S0Graph
    n::Integer
    sig::AlgebraShape
    S::Subspace{Complex{Float64}, 2}
    S0::Subspace{Complex{Float64}, 2}
    S1::Subspace{Complex{Float64}, 2} # commutant of S0

    function S0Graph(sig::AlgebraShape, S::Subspace{Complex{Float64}, 2})
        S0, S1 = create_S0_S1(sig)
        n = shape(S0)[1]
        S == S' || throw(DomainError("S is not an S0-graph"))
        S0 in S || throw(DomainError("S is not an S0-graph"))
        (S == S0 * S * S0) || throw(DomainError("S is not an S0-graph"))
        return new(n, sig, S, S0, S1)
    end

    function S0Graph(g::AbstractGraph)
        n = nv(g)
        S0 = Subspace([ (x=zeros(Complex{Float64}, n,n); x[i,i]=1; x) for i in 1:n ])
        S  = Subspace([ (x=zeros(Complex{Float64}, n,n); x[src(e),dst(e)]=1; x) for e in edges(g) ])
        S = S + S' + S0
        return S0Graph(ones(Int64, n, 2), S)
    end
end

function show(io::IO, g::S0Graph)
    print(io, "S0Graph{S0=$(g.sig) S=$(g.S)}")
end

vertex_graph(g::S0Graph) = S0Graph(g.sig, g.S0)

forget_S0(g::S0Graph) = S0Graph([1 g.n], g.S)

function random_S0Graph(sig::AlgebraShape)
    S0, S1 = create_S0_S1(sig)

    num_blocks = size(sig, 1)
    function block(col, row)
        da_col, dy_col = sig[col,:]
        da_row, dy_row = sig[row,:]
        ds = Integer(round(sqrt(dy_row * dy_col) / 2.0))
        F = full_subspace((da_row, da_col))
        if row == col
            R = random_hermitian_subspace(ds, dy_row)
        elseif row > col
            R = random_subspace(ds, (dy_row, dy_col))
        else
            R = empty_subspace((dy_row, dy_col))
        end
        return kron(F, R)
    end
    blocks = Array{Subspace{Complex{Float64}, 2}, 2}([
        block(col, row)
        for col in 1:num_blocks, row in 1:num_blocks
   ])

    S = hvcat(num_blocks, blocks...)
    S |= S'
    S |= S0

    return S0Graph(sig, S)
end

function empty_S0Graph(sig::AlgebraShape)
    S0, S1 = create_S0_S1(sig)
    return S0Graph(sig, S0)
end

complement(g::S0Graph) = S0Graph(g.sig, perp(g.S) | g.S0)

function ==(a::S0Graph, b::S0Graph)
    return a.sig == b.sig && a.S == b.S
end

function get_block_spaces(g::S0Graph)
    num_blocks = size(g.sig, 1)
    da_sizes = g.sig[:,1]
    dy_sizes = g.sig[:,2]
    n_sizes = da_sizes .* dy_sizes

    blkspaces = Array{Subspace{Complex{Float64}}, 2}(undef, num_blocks, num_blocks)
    offseti = 0
    for blki in 1:num_blocks
        offsetj = 0
        for blkj in 1:num_blocks
            #@show [blki, blkj, offseti, offsetj]
            blkbasis = Array{Array{Complex{Float64}, 2}, 1}()
            for m in each_basis_element(g.S)
                blk = m[1+offseti:dy_sizes[blki]+offseti, 1+offsetj:dy_sizes[blkj]+offsetj]
                push!(blkbasis, blk)
            end
            blkspaces[blki, blkj] = Subspace(blkbasis)
            #println(blkspaces[blki, blkj])
            offsetj += n_sizes[blkj]
        end
        @assert offsetj == shape(g.S)[2]
        offseti += n_sizes[blki]
    end
    @assert offseti == shape(g.S)[1]

    return blkspaces
end

function from_block_spaces(sig::AlgebraShape, blkspaces::Array{Subspace{Complex{Float64}}, 2})
    S0, S1 = NoncommutativeGraphs.create_S0_S1(sig)

    num_blocks = size(sig, 1)
    function block(col, row)
        da_col, dy_col = sig[col,:]
        da_row, dy_row = sig[row,:]
        ds = Integer(round(sqrt(dy_row * dy_col) / 2.0))
        F = full_subspace((da_row, da_col))
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

    J = cat([
        cat([ kron(eye(dA), basis_mat(dY, i)) for i in 1:dY^2 ]..., dims=3)
        for (dA, dY) in eachrow(sig)
    ]..., dims=(1,2,3))

    return reshape(J, (n^2, size(J)[3]))
end

function random_S1_unitary(sig::AlgebraShape)
    return cat([
        kron(eye(dA), rand(Haar(2), dY))
        for (dA, dY) in eachrow(sig)
    ]..., dims=(1,2))
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

function dsw_schur(g::S0Graph)
    n = shape(g.S)[1]

    function make_var(m)
        Z = ComplexVariable(n, n)
        add_constraint!(Z, Z == Z') # FIXME it'd be nice to have HermitianVariable
        return kron(m, Z)
    end
    Z = sum(make_var(m) for m in hermitian_basis(g.S))

    λ = Variable()
    wt = partialtrace(Z, 1, [n; n])
    wv = reshape(wt, n*n, 1)

    add_constraint!(λ, [ λ  wv' ; wv  Z ] ⪰ 0)

    return (λ=λ, w=transpose(wt), Z=Z)
end

# Like dsw_schur except much faster (when S0 != I), but w is constrained to S1.
function dsw_schur2(g::S0Graph)
    da_sizes = g.sig[:,1]
    dy_sizes = g.sig[:,2]
    n_sizes = da_sizes .* dy_sizes
    num_blocks = size(g.sig)[1]
    d = sum(dy_sizes .^ 2)

    blkspaces = get_block_spaces(g)
    Z = ComplexVariable(d, d)

    blkw = []
    offseti = 0
    for blki in 1:num_blocks
        offsetj = 0
        ni = dy_sizes[blki]^2
        for blkj in 1:num_blocks
            nj = dy_sizes[blkj]^2
            #@show [ni, nj]
            #@show [1+offseti:ni+offseti, 1+offsetj:nj+offsetj]
            blkZ = Z[1+offseti:ni+offseti, 1+offsetj:nj+offsetj]
            #@show size(blkZ)
            if blkj <= blki
                p = kron(
                    perp(blkspaces[blki, blkj]),
                    full_subspace((dy_sizes[blki], dy_sizes[blkj])))
                for m in each_basis_element(p)
                    add_constraint!(Z, tr(m' * blkZ) == 0)
                end
            end
            if blkj == blki
                wi = partialtrace(blkZ, 1, [dy_sizes[blki], dy_sizes[blkj]])
                push!(blkw, wi)
            end
            offsetj += nj
        end
        #@show [offsetj, d]
        @assert offsetj == d
        offseti += ni
    end
    @assert offseti == d

    #@show [ size(wi) for wi in blkw ]

    λ = Variable()
    wv = vcat([ reshape(wi, dy_sizes[i]^2, 1) for (i,wi) in enumerate(blkw) ]...)
    #@show size(wv)
    #@show size(Z)

    add_constraint!(λ, [ λ  wv' ; wv  Z ] ⪰ 0)

    wt = diagcat([ kron(eye(da_sizes[i]), wi) for (i,wi) in enumerate(blkw) ]...)
    #@show size(wt)

    return (λ=λ, w=transpose(wt), Z=Z)
end

function dsw(g::S0Graph, w::AbstractArray{<:Number, 2}; use_diag_optimization=true, eps=1e-6)
    if use_diag_optimization
        λ, x, Z = dsw_schur2(g)
    else
        λ, x, Z = dsw_schur(g)
    end

    problem = minimize(λ, [x ⪰ w])
    solve!(problem, () -> SCS.Optimizer(verbose=0, eps=eps))
    return (λ=problem.optval, x=Hermitian(evaluate(x)), Z=Hermitian(evaluate(Z)))
end

function dsw_antiblocker(g::S0Graph, w::AbstractArray{<:Number, 2}; use_diag_optimization=true, eps=1e-6)
    if use_diag_optimization
        # max{ <w,x> : Ψ(S, x) ⪯ y, ϑ(S, y) ≤ 1, y ∈ S1 }
        # equal to:
        # max{ dsw(S0, √y * w * √y) : dsw(complement(S), y) <= 1 }
        λ, y, Z = dsw_schur2(g)
        x = HermitianSemidefinite(g.n, g.n)
        problem = maximize(real(tr(w * x')), [ λ <= 1, Ψ(g, x) == y ])
        solve!(problem, () -> SCS.Optimizer(verbose=0, eps=eps))
        return (λ=problem.optval, x=Hermitian(evaluate(x)), y=Hermitian(evaluate(y)), Z=Hermitian(evaluate(Z)))
    else
        λ, x, Z = dsw_schur(g)
        problem = maximize(real(tr(w * x')), [ λ <= 1 ])
        solve!(problem, () -> SCS.Optimizer(verbose=0, eps=eps))
        return (λ=problem.optval, x=Hermitian(evaluate(x)), Z=Hermitian(evaluate(Z)))
    end
end

end
