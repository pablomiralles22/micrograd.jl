
include("binary_ops.jl")

############## GENERAL

mutable struct BinaryOpTensor{T<:Real, N, M, K, OpType<:BinaryOp{T}} <: BaseDifferentiableTensor{T, K}
    val::Array{T, N}
    grad::Array{T, N}
    child1::Tensor{T, N}
    child2::Tensor{T, M}
    function BinaryOpTensor{T, N, M, K, OpType}(val::Array{T, N}, grad::Array{T, N}, child1::Tensor{T, N}, child2::Tensor{T, M}) where {T <: Real, N, M, K, OpType<:BinaryOp{T}}
        @assert N isa Integer
        @assert M isa Integer
        @assert K isa Integer
        @assert K == max(N, M)
        new{T, N, M, K, OpType}(val, grad, child1, child2)
    end
end

function BinaryOpTensor{T, N, M, K, OpType}(child1::Tensor{T, N}, child2::Tensor{T, M}) where {T <: Real, N, M, K, OpType<:BinaryOp{T}}
    val = op_apply(OpType, child1.val, child2.val);
    grad = zeros(T, size(val));
    return BinaryOpTensor{T, N, M, K, OpType}(val, grad, child1, child2);
end

############ PRODUCT

import Base: *

function op_apply(::Type{MultOp{T}}, X::Array{T, N}, Y::Array{T, M}) where {T<:Real, N, M}
    return Broadcast.broadcast(*, X, Y);
end

function backward(tensor::BinaryOpTensor{T, N, M, K, MultOp{T}}) where {T<:Real, N, M, K}
    dx = similar(tensor.child1.val);
    sum!(dx, Broadcast.broadcast(*, tensor.child2.val, tensor.grad));
    update_gradient(tensor.child1, dx);

    dy = similar(tensor.child2.val);
    sum!(dy, Broadcast.broadcast(*, tensor.child1.val, tensor.grad));
    update_gradient(tensor.child2, dy);
end

*(X::Tensor{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = BinaryOpTensor{T, N, M, max(N, M), MultOp{Float64}}(X, Y)
*(X::Tensor{T, N}, Y::Array{T, M}) where{T<:Real, N, M} = X * ConstantTensor(Y)
*(X::Array{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = ConstantTensor(X) * Y
*(X::Tensor{T, N}, y::T) where{T<:Real, N} = X * [y]
*(x::T, Y::Tensor{T, M}) where{T<:Real, M} = [x] * Y

############ ADDITION

import Base: +

function op_apply(::Type{AddOp{T}}, X::Array{T, N}, Y::Array{T, M}) where {T<:Real, N, M}
    return Broadcast.broadcast(+, X, Y);
end

function backward(tensor::BinaryOpTensor{T, N, M, K, AddOp{T}}) where {T<:Real, N, M, K}
    dx = similar(tensor.child1.val);
    sum!(dx, tensor.grad);
    update_gradient(tensor.child1, dx);

    dy = similar(tensor.child2.val);
    sum!(dy, tensor.grad);
    update_gradient(tensor.child2, dy);
end

+(X::Tensor{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = BinaryOpTensor{T, N, M, max(N, M), AddOp{Float64}}(X, Y)
+(X::Tensor{T, N}, Y::Array{T, M}) where{T<:Real, N, M} = X + ConstantTensor(Y)
+(X::Array{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = ConstantTensor(X) + Y
+(X::Tensor{T, N}, y::T) where{T<:Real, N} = X + [y]
+(x::T, Y::Tensor{T, M}) where{T<:Real, M} = [x] + Y

