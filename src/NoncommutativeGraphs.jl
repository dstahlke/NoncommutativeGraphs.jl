module NoncommutativeGraphs

using Subspaces
using Convex, SCS, LinearAlgebra

export S0Graph
export random_S0Graph, complement, vertex_graph, forget_S0
export Ψ
export dsw_schur!, dsw_schur2!, dsw
export dsw_min_X_diag
export daw_antiblocker
export daw_antiblocker_S0

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
        kron(F, R)
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

complement(g::S0Graph) = S0Graph(g.sig, perp(g.S) | g.S0)

###############
### DSW solvers
###############

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
    # Convex.jl doesn't support this function
    #out = cat(blocks..., dims=(1,2))
    out = vcat([
        hcat([
            row == col ? blocks[row] : zeros(n_sizes[row], n_sizes[col])
            for col in 1:num_blocks
        ]...)
        for row in 1:num_blocks
    ]...)
    @assert size(out) == (n, n)
    return out
end

function dsw_schur!(constraints::Array{Constraint,1}, g::S0Graph)
    n = shape(g.S)[1]

    function make_var(m)
        # FIXME it'd be nice to have HermitianVariable
        Z = ComplexVariable(n, n)
        push!(constraints, Z == Z')
        return kron(m, Z)
    end
    Z = sum(make_var(m) for m in hermitian_basis(g.S))

    lambda = Variable()
    w = partialtrace(Z, 1, [n; n])
    wv = reshape(w, n*n, 1)

    push!(constraints, [ lambda  wv' ; wv  Z ] ⪰ 0)

    # FIXME maybe need transpose
    return lambda, w, Z
end

function dsw_schur2!(constraints::Array{Constraint,1}, g::S0Graph)
    lambda, x, Z = dsw_schur!(constraints, g)

    # FIXME should exploit symmetries to reduce degrees of freedom
    # (e.g. if S1=diags then Z is block diagonal)
    for m in each_basis_element(perp(g.S1))
        push!(constraints, tr(m' * x) == 0)
    end

    return lambda, x, Z
end

function dsw(g::S0Graph, w::AbstractArray{<:Number, 2})
    constraints = Array{Constraint,1}()
    lambda, x, Z = dsw_schur!(constraints, g)

    push!(constraints, x ⪰ w)
    problem = minimize(lambda, constraints)

    solve!(problem, () -> SCS.Optimizer(verbose=0))
    return problem.optval, x, Z
end

function dsw_min_X_diag(g::S0Graph, w::AbstractArray{<:Number, 2})
    constraints = Array{Constraint,1}()
    lambda, x, Z = dsw_schur2!(constraints, g)

    push!(constraints, x ⪰ w)
    problem = minimize(lambda, constraints)

    solve!(problem, () -> SCS.Optimizer(verbose=0))

    x = evaluate(x)
    xproj = projection(S1, x)
    println("proj err: ", norm(x - xproj))
    return problem.optval, xproj
end

function dsw_antiblocker(g::S0Graph, w::AbstractArray{<:Number, 2})
    constraints = Array{Constraint,1}()
    lambda, x, Z = dsw_schur!(constraints, g)

    push!(constraints, lambda <= 1)
    problem = maximize(real(tr(w * x')), constraints)

    solve!(problem, () -> SCS.Optimizer(verbose=0))
    return problem.optval, evaluate(x)
end

# max{ dsw(S0, √y * w * √y) : dsw(perp(S)+S0, y) <= 1 }
# We can assume (y in S1) because those are the extreme points.
# eq:max_WZ used to relate dsw(S0, .) to |Ψ(x)|
#function dsw_antiblocker_v3(g::S0Graph, w::AbstractArray{<:Number, 2})
#    n = g.n
#    da_sizes = g.sig[:,1]
#    dy_sizes = g.sig[:,2]
#
#    constraints = Array{Constraint,1}()
#    lambda, y, Z = dsw_schur2!(constraints, g)
#    push!(constraints, lambda <= 1)
#
#    q = HermitianSemidefinite(n)
#
#    k = 0
#    for (dai, dyi) in zip(da_sizes, dy_sizes)
#        ni = dai * dyi
#        TrAi_q = partialtrace(q[k+1:k+ni, k+1:k+ni], 1, [dai; dyi])
#        TrAi_y = partialtrace(y[k+1:k+ni, k+1:k+ni], 1, [dai; dyi])
#        push!(constraints, (ni * dai^-2 * TrAi_y) ⪰ TrAi_q)
#        k += ni
#    end
#    @assert k == n
#
#    problem = maximize(real(tr(w * q')), constraints)
#
#    solve!(problem, () -> SCS.Optimizer(verbose=0))
#
#    y = evaluate(y)
#    yproj = projection(g.S1, y)
#    println("proj err: ", norm(y - yproj))
#    return problem.optval, yproj, evaluate(q)
#end

# max{ <w,q> : Ψ(S, q) ⪯ y, ϑ(S, y) ≤ 1, y ∈ S1 }
# equal to:
# max{ dsw(S0, √y * w * √y) : dsw(perp(S)+S0, y) <= 1 }
function daw_antiblocker_S0(g::S0Graph, w::AbstractArray{<:Number, 2})
    n = g.n
    da_sizes = g.sig[:,1]
    dy_sizes = g.sig[:,2]

    constraints = Array{Constraint,1}()
    lambda, y, Z = dsw_schur2!(constraints, g)
    push!(constraints, lambda <= 1)

    q = HermitianSemidefinite(n)

    push!(constraints, Ψ(g, q) ⪯ y)

    problem = maximize(real(tr(w * q')), constraints)

    solve!(problem, () -> SCS.Optimizer(verbose=0))

    y = evaluate(y)
    yproj = projection(g.S1, y)
    println("proj err: ", norm(y - yproj))
    return problem.optval, yproj, evaluate(q)
end

end # module
