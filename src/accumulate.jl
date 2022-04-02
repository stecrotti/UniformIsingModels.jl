function accumulate_left!(L, h, β)
    N = length(h)
    L[0][0] = 1
    for k in 1:N
        for s in -(k-1):k-1
            L[k][s] = exp(β*h[k])*L[k-1][s-1] +
                       exp(-β*h[k])*L[k-1][s+1]
        end
        L[k][k] = exp(β*h[k])*L[k-1][k-1]
        L[k][-k] = exp(-β*h[k])*L[k-1][-k+1]
    end
    L
end
function accumulate_left(h, β)
    N = length(h)
    L = OffsetVector([fill(0.0, -N:N) for i in 0:N], 0:N)
    accumulate_left!(L, h, β)
end

function accumulate_right!(R, h, β)
    N = length(h)
    R[N+1][0] = 1
    for k in N:-1:1
        for s in -(N-k):(N-k)
            R[k][s] = exp(β*h[k])*R[k+1][s-1] +
                       exp(-β*h[k])*R[k+1][s+1]
        end
        R[k][N-k+1] = exp(β*h[k])*R[k+1][N-k]
        R[k][-(N-k+1)] = exp(-β*h[k])*R[k+1][-(N-k)]
    end
    R
end
function accumulate_right(h, β)
    N = length(h)
    R = OffsetVector([fill(0.0, -N:N) for i in 1:N+1], 1:N+1)
    accumulate_right!(R, h, β)
end

