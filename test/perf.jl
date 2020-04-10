# simple walk
# simd + inbounds -> same performance as Base.sum
function walk(a)
    acc = zero(eltype(a))
    @simd for i in CartesianIndices(a)
        @inbounds acc += a[i]
    end
    return acc
end

# simple walk, non-trivial box
function walk2(a)
    acc = zero(eltype(a))
    I = CartesianIndices((2:size(a,1)-1, 2:size(a,1)-1))
    @simd for i in I
        @inbounds acc += a[i]
    end
    return acc
end

# walk, inline accesses
# not slower than single access
function walk3(a)
    acc = zero(eltype(a))
    I = CartesianIndices((2:size(a,1)-1, 2:size(a,1)-1))
    @inbounds @simd for i in I
        acc += a[i + CartesianIndex(-1,0)]
        acc += a[i + CartesianIndex(0,0)]
        acc += a[i + CartesianIndex(1,0)]
    end
    return acc
end

# walk, multiple accesses
function walk4(a)
    acc = zero(eltype(a))
    I = CartesianIndices((2:size(a,1)-1, 2:size(a,1)-1))
    # both @simd macros are needed
    # switching loops (kernel on the outsidle) has bad performance, walks image
    # multiple times
    @inbounds @simd for i in I
        @simd for j in (-1,0,1)
            acc += a[i + CartesianIndex(j,0)]
        end
        # not getting simd'd if memory accesses are not adjacent
        #@simd for j in (-1,0,1)
        #    acc += a[i + CartesianIndex(0,j)]
        #end
    end
    return acc
end

# walk, multiple accesses, external kernel
# now performance depends on ki even though generated code is identical:
# ki = (1, 0, -1) -> same performance
# ki = (-1, 0, 1) -> slightly slower
function walk5(a, ki, kw)
    acc = zero(eltype(a))
    I = CartesianIndices((2:size(a,1)-1, 2:size(a,1)-1))
    @inbounds @simd for i in I
        @simd for j in ki
            acc += kw[j] * a[i + CartesianIndex(j,0)]
        end
    end
    return acc
end

# filter, cartesian kernel
function walk6(a, b)
    acc = zero(eltype(a))
    I = CartesianIndices((4:size(a,1)-3, 4:size(a,1)-3))
    @inbounds @simd for i in I
        @simd for j in (1, 2, 3)
            io = i + CartesianIndex(j, 0)
            # separate lines are faster
            acc += b[io]
            acc += a[io]
        end
    end
    return acc
end

function walk7(a, b)
    acc = zero(eltype(a))
    I = CartesianIndices((4:size(a,1)-3, 4:size(a,1)-3))
    @inbounds @simd for i in I
        @simd for j in 1:3
            io = i + CartesianIndex(j, 0)
            acc += a[io]
        end
    end
    return acc
end

function walk8(a, ki, kw)
    acc = zero(eltype(a))
    I = CartesianIndices((4:size(a,1)-3, 4:size(a,1)-3))
    @inbounds @simd for i in I
        @simd for j in CartesianIndices(kw)
            acc += kw[j] * a[i + j]
        end
    end
    return acc
end

function walk9(a, k)
    acc = zero(eltype(a))
    @inbounds @simd for i in CartesianIndices(axes(a, k))
        acc += a[k, i]
    end
    return acc
end

