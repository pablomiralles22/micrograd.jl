###################################################################
######################### UNARY OPS IMPL ##########################
###################################################################
using LoopVectorization

abstract type UnaryOp{T<:Real} end

###### EXPONENTIAL

abstract type ExpOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{ExpOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return exp.(val);
end

function dop_apply(::Type{ExpOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return exp.(val);
end


###### RELU

abstract type ReluOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{ReluOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return max.(val, T(0.0));
end

function dop_apply(::Type{ReluOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return convert.(T, val .> 0.0);
end

###### TANH

abstract type TanhOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{TanhOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return tanh.(val);
end

function dop_apply(::Type{TanhOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return 1 .- (tanh.(val) .^ 2);
end

###### LOG

abstract type LogOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{LogOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return log.(val);
end

function dop_apply(::Type{LogOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return 1.0 ./ val;
end


###### OPPOSITE

abstract type UnaryMinusOp{T<:Real} <: UnaryOp{T} end

function op_apply(::Type{UnaryMinusOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return -val;
end

function dop_apply(::Type{UnaryMinusOp{T}}, val::Array{T, N}) where {T, N}
    @tturbo return -ones(size(val))
end

###################################################################
######################### UNARY OP TENSOR #########################
###################################################################

mutable struct UnaryOpTensor{T<:Real, N, OpType<:UnaryOp{T}} <: BaseDifferentiableTensor{T, N}
    val:: Array{T, N}
    grad::Array{T, N}
    child::Tensor{T, N}
end

function backward!(tensor::UnaryOpTensor{T, N, OpType}) where {T, N, OpType}
    if !(tensor.child isa ConstantTensor)
        delta = tensor.grad .* dop_apply(OpType, tensor.child.val);
        update_gradient!(tensor.child, delta);
    end
end

function forward(::Type{OpType}, tensor::Tensor{T, N})::UnaryOpTensor{T, N, OpType} where {T, N, OpType}
    val = op_apply(OpType, tensor.val);
    grad = zeros(T, size(val));
    return UnaryOpTensor{T, N, OpType}(val, grad, tensor)
end


###################################################################
######################## OPERATION EXPORT #########################
###################################################################

import Base: exp, tanh, log, -

exp(tensor::Tensor{T, N}) where {T, N} = forward(ExpOp{T}, tensor)
relu(tensor::Tensor{T, N}) where {T, N} = forward(ReluOp{T}, tensor)
tanh(tensor::Tensor{T, N}) where {T, N} = forward(TanhOp{T}, tensor)
log(tensor::Tensor{T, N}) where {T, N} = forward(LogOp{T}, tensor)
-(tensor::Tensor{T, N}) where {T, N} = forward(UnaryMinusOp{T}, tensor)

export exp, relu, tanh, log, -
