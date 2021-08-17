import Distributions
using DynamicPPL
import Bijectors
using StatsFuns
using Turing
using LinearAlgebra
using Random


# value that we subtract from 1, such that we don't deal with numerical instabilities
const smallval = 1e-8
# easy reusable value for 1 min the small value above, because we use it a lot.
const onemin = 1.0 - smallval
const oneminsq = onemin^2

# This code is by no means solely my own, it is just for personal use because I had problems with Cholesky stuff, similar to what was mentioned here: https://discourse.julialang.org/t/multi-level-varying-slopes-with-two-clusters-cross-classification/62036
# and in other places as well. Hence I quickly copied snippets from here and there, such as the main distribution code which comes from the PR by Seth Axen: https://github.com/JuliaStats/Distributions.jl/pull/1339
# and a lot of the bijectors stuff is also copied from the CorrBijector from Bijectors.jl

# However, I had to come up with a lot of implementation details for e.g. the Bijector myself, and had to implement some additional methods that were required for it to work in Turing
# such as the DynamicPPL.reconstruct and vectorize, the Turing.Utilities.FlattenIterator etc.
# Much of this code has no guarantee that it works!! I was still encountering bugs myself, such as in FlattenIterator I return cholesky.factors, but that matrix contain NaNs in the part of the matrix which was not being used. For example if the matrix was LowerTriangular, the upper part could contain NaNs, which would be returned with the chains and cause errors when doing statistics like quantiles which are done by default in Turing.

# I also had to do very dirty stuff in the Bijectors _link_chol_lkj_2 function, with using `min` function to deal with numerical instabilities. Otherwise it would throw DomainErrors because the value inside the sqrt would be negative somehow. Please let me know if there is a better or more efficient way to deal with it. Because even if this isn't efficient, it is better than the whole sampling process crashing because of a DomainError




abstract type CholeskyVariate <: VariateForm end
const CholeskyDistribution{S<:ValueSupport}       = Distribution{CholeskyVariate,S}
"""
    LKJCholesky(d, η, uplo='L')
The `LKJCholesky` distribution of size ``d`` with shape parameter ``\\eta`` is a
distribution over `Cholesky` factors of ``d\\times d`` real correlation matrices
(positive-definite matrices with ones on the diagonal).
Variates or samples of the distribution are `LinearAlgebra.Cholesky` objects, as might
be returned by `F = cholesky(R)`, so that `Matrix(F) ≈ R` is a variate or sample of
[`LKJ`](@ref). `LKJCholesky` is more efficient than `LKJ` when working with the Cholesky
factors of correlation matrices.
The `uplo` parameter specifies in which triangle in `F` (either `'U'` for upper or `'L'`
for lower) the Cholesky factor should be stored when randomly generating samples. Set `uplo`
to `'U'` if the upper factor is desired to avoid allocating a copy when calling `F.U`.
See [`LKJ`](@ref) for more details.
External links
* Lewandowski D, Kurowicka D, Joe H.
  Generating random correlation matrices based on vines and extended onion method,
  Journal of Multivariate Analysis (2009), 100(9): 1989-2001
  doi: [10.1016/j.jmva.2009.04.008](https://doi.org/10.1016/j.jmva.2009.04.008)
"""
struct LKJCholesky{T <: Real, D <: Integer} <: Distribution{CholeskyVariate,Continuous}
    d::D
    η::T
    uplo::Char
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function LKJCholesky(d::Integer, _η::Real, _uplo::Union{Char,Symbol} = 'L'; check_args = true)
    if check_args
        d > 0 || throw(ArgumentError("Matrix dimension must be positive."))
        _η > 0 || throw(ArgumentError("Shape parameter must be positive."))
    end
    _logc0 = Distributions.lkj_logc0(d, _η)
    uplo = _char_uplo(_uplo)
    uplo ∈ ('U', 'L') || throw(ArgumentError("uplo must be 'U' or 'L'."))
    η, logc0 = promote(_η, _logc0)
    return LKJCholesky(d, η, uplo, logc0)
end

# adapted from LinearAlgebra.char_uplo
function _char_uplo(_uplo::Union{Symbol,Char})
    uplo = if _uplo === :U
        'U'
    elseif _uplo === :L
        'L'
    else
        _uplo
    end
    uplo ∈ ('U', 'L') && return uplo
    throw(ArgumentError("uplo argument must be either 'U' (upper) or 'L' (lower)"))
end

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

Base.show(io::IO, d::LKJCholesky) = Base.show(io, d, (:d, :η, :uplo))

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{LKJCholesky{T}}, d::LKJCholesky) where T <: Real
    return LKJCholesky{T, typeof(d.d)}(d.d, T(d.η), d.uplo, T(d.logc0))
end
function convert(::Type{LKJCholesky{T}}, d::Integer, η::Real, uplo::Char, logc0::Real) where T <: Real
    return LKJCholesky{T, typeof(d)}(d, T(η), uplo, T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

Distributions.dim(d::LKJCholesky) = d.d

function Base.size(d::LKJCholesky)
    p = dim(d)
    return (p, p)
end

function Distributions.insupport(d::LKJCholesky, R::Cholesky)
    p = dim(d)
    factors = R.factors
    (isreal(factors) && Base.size(factors, 1) == p) || return false
    iinds, jinds = axes(factors)
    # check that the diagonal of U'*U or L*L' is all ones
    @inbounds if R.uplo === 'U'
        for (j, jind) in enumerate(jinds)
            col_iinds = view(iinds, 1:j)
            # perhaps the tolerance should even be sqrt(smallval^2 + 2*smallval) in case I get errors still
            # but ingeral sqrt(smallval) seems good, because it should relate to the value we use in all the statements where we take the minimum between 1-tol and the actual value
            isapprox(sum(abs2(factors[iind, jind]) for iind in col_iinds), 1; rtol=sqrt(smallval)) || return false
        end
    else  # R.uplo === 'L'
        for (i, iind) in enumerate(iinds)
            row_jinds = view(jinds, 1:i)
            isapprox(sum(abs2(factors[iind, jind]) for jind in row_jinds), 1; rtol=sqrt(smallval)) || return false
        end
    end
    return true
end

function Distributions.mode(d::LKJCholesky)
    factors = Matrix{eltype(d)}(I, size(d))
    return Cholesky(factors, d.uplo, 0)
end

Distributions.params(d::LKJCholesky) = (d.d, d.η, d.uplo)

@inline Distributions.partype(::LKJCholesky{T}) where {T <: Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function Distributions.logkernel(d::LKJCholesky, R::Cholesky)
    factors = R.factors
    p, η = params(d)
    c = p + 2(η - 1)
    p == 1 && return c * log(first(factors))
    logp = sum(Iterators.drop(enumerate(diagind(factors)), 1)) do (i, di) 
        return (c - i) * log(factors[di])
    end
    return logp
end

function Distributions.logpdf(d::LKJCholesky, R::Cholesky)
    if !Distributions.insupport(d, R)
        throw(ArgumentError("Provided point is not in the support."))
    end
    return Distributions._logpdf(d, R)
end

Distributions._logpdf(d::LKJCholesky, R::Cholesky) = Distributions.logkernel(d, R) + d.logc0

Distributions.pdf(d::LKJCholesky, R::Cholesky) = exp(Distributions.logpdf(d, R))

Distributions.loglikelihood(d::LKJCholesky, R::Cholesky) = Distributions.logpdf(d, R)
function Distributions.loglikelihood(d::LKJCholesky, Rs::AbstractArray{<:Cholesky})
    return sum(R -> Distributions.logpdf(d, R), Rs)
end


#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function Distributions.rand(rng::AbstractRNG, d::LKJCholesky)
    factors = Matrix{eltype(d)}(undef, size(d))
    # factors = zeros(eltype(d), size(d))
    R = Cholesky(factors, d.uplo, 0)
    return _lkj_cholesky_onion_sampler!(rng, d, R)
end
function Distributions.rand(rng::AbstractRNG, d::LKJCholesky, dims::Dims)
    p = dim(d)
    uplo = d.uplo
    T = eltype(d)
    TM = Matrix{T}
    Rs = Array{Cholesky{T,TM}}(undef, dims)
    for i in eachindex(Rs)
        factors = TM(undef, p, p)
        Rs[i] = R = Cholesky(factors, uplo, 0)
        _lkj_cholesky_onion_sampler!(rng, d, R)
    end
    return Rs
end

Distributions.rand!(d::LKJCholesky, R::Cholesky) = Distributions.rand!(GLOBAL_RNG, d, R)
Distributions.rand!(rng::AbstractRNG, d::LKJCholesky, R::Cholesky) = _lkj_cholesky_onion_sampler!(rng, d, R)

function Distributions.rand!(rng::AbstractRNG, d::LKJCholesky, Rs::AbstractArray{<:Cholesky{T,TM}}, allocate::Bool) where {T,TM}
    p = dim(d)
    uplo = d.uplo
    if allocate
        for i in eachindex(Rs)
            Rs[i] = _lkj_cholesky_onion_sampler!(rng, d, Cholesky(TM(undef, p, p), uplo, 0))
        end
    else
        for i in eachindex(Rs)
            _lkj_cholesky_onion_sampler!(rng, d, Rs[i])
        end
    end
    return Rs
end
function Distributions.rand!(rng::AbstractRNG, d::LKJCholesky, Rs::AbstractArray{<:Cholesky})
    allocate = any(!isassigned(Rs, i) for i in eachindex(Rs)) || any(R -> size(R, 1) != dim(d), Rs)
    return rand!(rng, d, Rs, allocate)
end

#
# onion method
#

function _lkj_cholesky_onion_sampler!(rng::AbstractRNG, d::LKJCholesky, R::Cholesky)
    TTri = R.uplo === 'U' ? UpperTriangular : LowerTriangular
    _lkj_cholesky_onion_tri!(rng, R.factors, d.d, d.η, TTri)
    return R
end


function _lkj_cholesky_onion_tri!(
    rng::AbstractRNG,
    A::AbstractMatrix,
    d::Integer,
    η::Real,
    ::Type{TTri},
) where {TTri<:LinearAlgebra.AbstractTriangular}
    # Section 3.2 in LKJ (2009 JMA)
    # reformulated to incrementally construct Cholesky factor as mentioned in Section 5
    # equivalent steps in algorithm in reference are marked.
    @assert Base.size(A) == (d, d)
    A[1, 1] = 1
    d > 1 || return R
    β = η + (d - 2)//2
    #  1. Initialization
    w0 = 2 * rand(rng, Beta(β, β)) - 1
    @inbounds if TTri <: LowerTriangular
        A[2, 1] = w0
    else
        A[1, 2] = w0
    end
    @inbounds A[2, 2] = sqrt(1 - w0^2)
    #  2. Loop, each iteration k adds row/column k+1
    for k in 2:(d - 1)
        #  (a)
        β -= 1//2
        #  (b)
        y = rand(rng, Beta(k//2, β))
        #  (c)-(e)
        # w is directionally uniform vector of length √y
        @inbounds w = @views TTri <: LowerTriangular ? A[k + 1, 1:k] : A[1:k, k + 1]
        randn!(rng, w)
        rmul!(w, sqrt(y) / norm(w))
        # normalize so new row/column has unit norm
        @inbounds A[k + 1, k + 1] = sqrt(1 - y)
    end
    #  3.
    return A
end


### ---------------- DynamicPPL stuff -----------------------------

DynamicPPL.vectorize(d::LKJCholesky, r::Cholesky) =  copy(vec( r.factors ))


function DynamicPPL.reconstruct(d::LKJCholesky, val::AbstractVector)
    return  LinearAlgebra.Cholesky(reshape(copy(val), size(d)), 'L', 0)
end



### ---------------- Bijector stuff -----------------------------

struct LKJCholBijector <: Bijector{2} end

function (b::LKJCholBijector)(x::Cholesky)
    return LinearAlgebra.Cholesky(Array(_link_w_lkj_chol(x.U.data)') + zero(x.U.data), 'L', 0)
end
   
(b::LKJCholBijector)(X::AbstractArray{<:Cholesky}) = map(b, X)

function (ib::Inverse{<:LKJCholBijector})(y::Cholesky) 
    return LinearAlgebra.Cholesky(Array(_inv_link_w_lkj_chol(y.U.data)'), 'L', 0)
end

(ib::Inverse{<:LKJCholBijector})(Y::AbstractArray{<:Cholesky}) = map(ib, Y)


function Bijectors.logabsdetjac(::Inverse{LKJCholBijector}, y::Cholesky)
    K = LinearAlgebra.checksquare(y)
    
    result = float(zero(eltype(y)))
    for j in 2:K, i in 1:(j - 1)
        @inbounds abs_y_i_j = abs((y.L)[i, j])
        result += (K - i + 1) * (logtwo - (abs_y_i_j + StatsFuns.log1pexp(-2 * abs_y_i_j)))
    end
    
    return result
end
function Bijectors.logabsdetjac(b::LKJCholBijector, X::Cholesky)
    #=
    It may be more efficient if we can use un-contraint value to prevent call of b
    It's recommended to directly call 
    `logabsdetjac(::Inverse{CorrBijector}, y::AbstractMatrix{<:Real})`
    if possible.
    =#
    return -Bijectors.logabsdetjac(inv(b), (b(X))) 
end
function Bijectors.logabsdetjac(b::LKJCholBijector, X::AbstractArray{<:Cholesky})
    return Bijectors.mapvcat(X) do x
        Bijectors.logabsdetjac(b, x)
    end
end




function _inv_link_w_lkj_chol(y)
    K = LinearAlgebra.checksquare(y)
    w = zero(y)
    
    @inbounds for j in 1:K
        w[1, j] = 1
        for i in 2:j
            z = tanh(y[i-1, j])
            tmp = w[i-1, j]

            w[i-1, j] = z * tmp

            w[i, j] = tmp * sqrt(1 - min(typeof(z)(oneminsq), z^2))
        end
        for i in (j+1):K
            w[i, j] = 0
        end
    end
    return w
end


function _link_w_lkj_chol(w)
    K = LinearAlgebra.checksquare(w)

    z = similar(w) # z is also UpperTriangular. 
    # Some zero filling can be avoided. Though diagnoal is still needed to be filled with zero.

    # This block can't be integrated with loop below, because w[1,1] != 0.
    @inbounds z[1, 1] = 0

    @inbounds for j=2:K
        tmp_w = max(typeof(w[1,j])(-onemin), min(typeof(w[1,j])(onemin), w[1, j]))
        z[1, j] = atanh(tmp_w)
        tmp = sqrt(1 - tmp_w^2)
        for i in 2:(j - 1)
            p = w[i, j] / tmp
            p = max(typeof(p)(-onemin), min(typeof(p)(onemin), p))

            tmp *= sqrt(1 - p^2)
            z[i, j] = atanh(p)
        end
        z[j, j] = 0
    end
    return z
end





Bijectors.bijector(d::LKJCholesky) = LKJCholBijector()


### ---------------- Other stuff -----------------------------


# Base.length(::Cholesky) = 1

Turing.Utilities.FlattenIterator(name, value::Cholesky) = Turing.Utilities.FlattenIterator(Symbol(name), Array(value.L))



	
using ReverseDiff


_inv_link_w_lkj_chol(y::AbstractArray{<:ReverseDiff.TrackedReal}) = ReverseDiff.track(_inv_link_w_lkj_chol, y)

ReverseDiff.@grad function _inv_link_w_lkj_chol(y_tracked::AbstractArray)
    y = ReverseDiff.value(y_tracked)
    
    K = LinearAlgebra.checksquare(y)
    w = similar(y)

    z_mat = similar(y) # cache for adjoint
    tmp_mat = similar(y)
    
    @inbounds for j in 1:K
        w[1, j] = 1
        for i in 2:j
            z = tanh(y[i-1, j])
            tmp = w[i-1, j]
            w[i-1, j] = z * tmp

            z_mat[i, j] = z
            tmp_mat[i, j] = tmp
            
            w[i, j] = tmp * sqrt(1 - min(typeof(z)(oneminsq), z^2))
        end
        for i in (j+1):K
            w[i, j] = 0
        end
    end

    function pullback_inv_link_chol_lkj(Δw)
        LinearAlgebra.checksquare(Δw)

        Δy = zero(y)

        @inbounds for j in 1:K
            Δtmp = Δw[j,j]
            for i in j:-1:2
                z_mat_min_sq = min(typeof(z_mat[i, j])(oneminsq), z_mat[i, j]^2)
                Δz = Δw[i-1, j] * tmp_mat[i, j] - Δtmp * tmp_mat[i, j] / sqrt(1 - z_mat_min_sq) * z_mat[i, j]
                Δy[i-1, j] = Δz / cosh(y[i-1, j])^2
                Δtmp = Δw[i-1, j] * z_mat[i, j] + Δtmp * sqrt(1 - z_mat_min_sq)
            end
        end
        
        return (Δy,)
    end

    return w, pullback_inv_link_chol_lkj
end

_link_w_lkj_chol(w::AbstractArray{<:ReverseDiff.TrackedReal}) = ReverseDiff.track(_link_w_lkj_chol, w)
ReverseDiff.@grad function _link_w_lkj_chol(w_tracked::AbstractArray)
    w = ReverseDiff.value(w_tracked)
    K = LinearAlgebra.checksquare(w)
    
    z = similar(w)

    @inbounds z[1, 1] = 0

    tmp_mat = similar(w) # cache for pullback.

    @inbounds for j=2:K
        tmp_w = max(typeof(w[1, j])(-onemin), min(typeof(w[1, j])(onemin), w[1, j]))
        z[1, j] = atanh(tmp_w)
        tmp = sqrt(1 - tmp_w^2)
        tmp_mat[1, j] = tmp
        for i in 2:(j - 1)
            p = w[i, j] / tmp
            p = max(typeof(p)(-onemin), min(typeof(p)(onemin), p))

            tmp *= sqrt(1 - p^2)
            tmp_mat[i, j] = tmp
            z[i, j] = atanh(p)
        end
        z[j, j] = 0
    end

    function pullback_link_chol_lkj(Δz)
        LinearAlgebra.checksquare(Δz)

        Δw = similar(w)

        @inbounds Δw[1,1] = zero(eltype(Δz))

        @inbounds for j=2:K
            Δw[j, j] = 0
            Δtmp = zero(eltype(Δz)) # Δtmp_mat[j-1,j]
            for i in (j-1):-1:2
                p = w[i, j] / tmp_mat[i-1, j]
                pmin = max(typeof(p)(-onemin), min(typeof(p)(onemin), p))


                ftmp = sqrt(1 - pmin^2)
                d_ftmp_p = -p / ftmp
                d_p_tmp = -w[i,j] / tmp_mat[i-1, j]^2

                Δp = Δz[i,j] / (1-p^2) + Δtmp * tmp_mat[i-1, j] * d_ftmp_p
                Δw[i, j] = Δp / tmp_mat[i-1, j]
                Δtmp = Δp * d_p_tmp + Δtmp * ftmp # update to "previous" Δtmp
            end
            Δw[1, j] = Δz[1, j] / (1-w[1,j]^2) - Δtmp / sqrt(1 - min(typeof(w[1, j])(oneminsq), w[1, j]^2)) * w[1,j]
        end

        return (Δw,)
    end

    return z, pullback_link_chol_lkj
end
