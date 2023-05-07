abstract type UnaryOp{T<:Real} end


###### EXPONENTIAL

abstract type ExpOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{ExpOp{T}}, val::Array{T, N}) where {T, N}
    return exp.(val);
end

function dop_apply(::Type{ExpOp{T}}, val::Array{T, N}) where {T, N}
    return exp.(val);
end


###### RELU

abstract type ReluOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{ReluOp{T}}, val::Array{T, N}) where {T, N}
    return max.(val, T(0.0));
end

function dop_apply(::Type{ReluOp{T}}, val::Array{T, N}) where {T, N}
    return convert.(T, val .> 0.0);
end

###### TANH

abstract type TanhOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{TanhOp{T}}, val::Array{T, N}) where {T, N}
    return tanh.(val);
end

function dop_apply(::Type{TanhOp{T}}, val::Array{T, N}) where {T, N}
    return 1 .- (tanh.(val) .^ 2);
end

###### LOG

abstract type LogOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{LogOp{T}}, val::Array{T, N}) where {T, N}
    return log.(val);
end

function dop_apply(::Type{LogOp{T}}, val::Array{T, N}) where {T, N}
    return 1.0 ./ val;
end


###### OPPOSITE

abstract type UnaryMinusOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{UnaryMinusOp{T}}, val::Array{T, N}) where {T, N}
    return -val;
end

function dop_apply(::Type{UnaryMinusOp{T}}, val::Array{T, N}) where {T, N}
    return -ones(size(val))
end
