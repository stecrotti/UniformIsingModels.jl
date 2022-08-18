import FastGaussQuadrature
import LinearAlgebra: dot

function exp_value(f; npoints=10^2, 
        t = FastGaussQuadrature.gausshermite(npoints))
    x, w = t
    dot(w, f.(x))
end

function f_fp(m, J, Δ, β; npoints=5*10^2, 
        t = FastGaussQuadrature.gausshermite(npoints))
    sh(x) = sinh(β*(x*sqrt(Δ)+J*m))
    ch(x) = cosh(β*(x*sqrt(Δ)+J*m))
    num = exp_value(sh; t)
    den = exp_value(ch; t)
    num / den
end

function solve_fp(J, Δ, β; niters=10^2, tol=1e-8, damp=0.0,
        m0=0.5, npoints=10^2,
        cb = (m, m_new, it, tol) -> isapprox(m, m_new, rtol=tol))
    x, w = FastGaussQuadrature.gausshermite(npoints)
    m = m0
    for it in 1:niters
        m_new = f_fp(m, J, Δ, β; t=(x,w))
        cb(m, m_new, it, tol) && return m
        m = damp*m + (1-damp)*m_new
    end
    m
end

J = 1.0
β = 1.0
Δ = 1.0

m = 0.0
f_fp(m, J, Δ, β)

function free_entropy(m, J, Δ, β)
    ch(x) = cosh(β*(x*sqrt(Δ)+J*m))
    entr = - 2*log(exp_value(ch))
    ener = - β*J*m^2/2
    ener - entr
end