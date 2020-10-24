module NoncommutativeGraphs

using Subspaces
using Convex, SCS, LinearAlgebra

export NoncommutativeGraph
export dsw_schur!, dsw, dsw_antiblocker
export dsw_min_X_diag

NoncommutativeGraph = Subspace{<:Number, 2}

function dsw_schur!(constraints::Array{Constraint,1}, S::NoncommutativeGraph)
    n = shape(S)[1]

    function make_var(m)
        # FIXME it'd be nice to have HermitianVariable
        Z = ComplexVariable(n, n)
        push!(constraints, Z == Z')
        return kron(m, Z)
    end
    Z = sum(make_var(m) for m in hermitian_basis(S))

    lambda = Variable()
    w = partialtrace(Z, 1, [n; n])
    wv = reshape(w, n*n, 1)

    push!(constraints, [ lambda  wv' ; wv  Z ] ⪰ 0)

    # FIXME maybe need transpose
    return lambda, w, Z
end

# S1 is commutant of S0
function dsw_schur!(constraints::Array{Constraint,1}, S::NoncommutativeGraph, S1::NoncommutativeGraph)
    lambda, x, Z = dsw_schur!(constraints, S)

    # FIXME should exploit symmetries to reduce degrees of freedom
    # (e.g. if S1=diags then Z is block diagonal)
    for m in each_basis_element(perp(S1))
        push!(constraints, tr(m' * x) == 0)
    end

    return lambda, x, Z
end

function dsw(S::NoncommutativeGraph, w::AbstractArray{<:Number, 2})
    constraints = Array{Constraint,1}()
    lambda, x, Z = dsw_schur!(constraints, S)

    push!(constraints, x ⪰ w)
    problem = minimize(lambda, constraints)

    solve!(problem, () -> SCS.Optimizer(verbose=0))
    return problem.optval, x, Z
end

function dsw_antiblocker(S::NoncommutativeGraph, w::AbstractArray{<:Number, 2})
    constraints = Array{Constraint,1}()
    lambda, x, Z = dsw_schur!(constraints, S)

    push!(constraints, lambda <= 1)
    problem = maximize(real(tr(w * x')), constraints)

    solve!(problem, () -> SCS.Optimizer(verbose=0))
    return problem.optval, evaluate(x)
end

function dsw_min_X_diag(S::NoncommutativeGraph, S1::NoncommutativeGraph, w::AbstractArray{<:Number, 2})
    constraints = Array{Constraint,1}()
    lambda, x, Z = dsw_schur!(constraints, S, S1)

    push!(constraints, x ⪰ w)
    problem = minimize(lambda, constraints)

    solve!(problem, () -> SCS.Optimizer(verbose=0))

    x = evaluate(x)
    xproj = projection(S1, x)
    println("proj err: ", norm(x - xproj))
    return problem.optval, xproj
end

end # module
