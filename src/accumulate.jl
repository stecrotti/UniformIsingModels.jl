function accumulate_left!(L, h, β)
    N = length(h)
    L[0][0] = 0.0
    for k in 1:N
        for s in -(k-1):k-1
            L[k][s] = logaddexp( β*h[k] + L[k-1][s-1],
                                -β*h[k] + L[k-1][s+1] )
        end
        L[k][k] = β*h[k] + L[k-1][k-1]
        L[k][-k] = -β*h[k] + L[k-1][-k+1]
    end
    L
end
function accumulate_left(h, β)
    N = length(h)
    L = OffsetVector([fill(-Inf, -N:N) for i in 0:N], 0:N)
    accumulate_left!(L, h, β)
end

function accumulate_d_left!(L, dLdβ, h, β)
    N = length(h)
    L[0] .= -Inf; L[0][0] = 0.0; dLdβ[0] .= 0.0
    for k in 1:N
        for s in -(k-1):k-1
            L[k][s] = logaddexp( β*h[k] + L[k-1][s-1],
                                -β*h[k] + L[k-1][s+1] )
            dLdβ[k][s] = (+h[k] + dLdβ[k-1][s-1])*exp(+β*h[k]+L[k-1][s-1]) + 
                       (-h[k] + dLdβ[k-1][s+1])*exp(-β*h[k]+L[k-1][s+1])
            dLdβ[k][s] = dLdβ[k][s] / exp(L[k][s])
            isnan(dLdβ[k][s]) && (dLdβ[k][s] = 0.0)
        end
        L[k][k] = β*h[k] + L[k-1][k-1]
        L[k][-k] = -β*h[k] + L[k-1][-k+1]
        dLdβ[k][k] = h[k] + dLdβ[k-1][k-1]
        dLdβ[k][-k] = -h[k] + dLdβ[k-1][-k+1]
    end
    L, dLdβ
end
function accumulate_d_left(h, β)
    N = length(h)
    L = OffsetVector([fill(-Inf, -N:N) for i in 0:N], 0:N)
    dLdβ = OffsetVector([fill(0.0, -N:N) for i in 0:N], 0:N)
    accumulate_d_left!(L, dLdβ, h, β)
end

function accumulate_right!(R, h, β)
    N = length(h)
    R[N+1][0] = 0.0
    for k in N:-1:1
        for s in -(N-k):(N-k)
            R[k][s] = logaddexp( β*h[k] + R[k+1][s-1],
                                -β*h[k] + R[k+1][s+1] )
        end
        R[k][N-k+1] = β*h[k] + R[k+1][N-k]
        R[k][-(N-k+1)] = -β*h[k] + R[k+1][-(N-k)]
    end
    R
end
function accumulate_right(h, β)
    N = length(h)
    R = OffsetVector([fill(-Inf, -N:N) for i in 1:N+1], 1:N+1)
    accumulate_right!(R, h, β)
end

function accumulate_middle!(M, h, β)
    N = length(h)
    for i in 1:N
        for s in (-1,1)
            M[i,i][s] = β*h[i]*s
        end
        for j in i+1:N
            for s in -(j-i+1):j-i+1
                M[i,j][s] = logaddexp( β*h[j] + M[i,j-1][s-1], 
                                      -β*h[j] + M[i,j-1][s+1] )
            end
        end
    end
    M
end
function accumulate_middle(h, β)
    N = length(h)
    M = [fill(-Inf, -(N+1):(N+1)) for i in 1:N, j in 1:N]
    accumulate_middle!(M, h, β)
end