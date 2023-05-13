
###################################################################
######################## BINARY OP TYPES ##########################
###################################################################

abstract type BinaryOp{T<:Real} end

abstract type MultOp{T<:Real} <: BinaryOp{T} end
abstract type AddOp{T<:Real} <: BinaryOp{T} end
abstract type PowOp{T<:Real} <: BinaryOp{T} end


###################################################################
######################## TENSOR STRUCT ############################
###################################################################

mutable struct BinaryOpTensor{T<:Real, N, M, K, OpType<:BinaryOp{T}} <: BaseDifferentiableTensor{T, K}
    val::Array{T, K}
    grad::Array{T, K}
    child1::Tensor{T, N}
    child2::Tensor{T, M}
    function BinaryOpTensor{T, N, M, K, OpType}(val::Array{T, K}, grad::Array{T, K}, child1::Tensor{T, N}, child2::Tensor{T, M}) where {T <: Real, N, M, K, OpType<:BinaryOp{T}}
        @assert N isa Integer
        @assert M isa Integer
        @assert K isa Integer
        @assert K == max(N, M)
        new{T, N, M, K, OpType}(val, grad, child1, child2)
    end
end

function BinaryOpTensor{T, N, M, K, OpType}(child1::Tensor{T, N}, child2::Tensor{T, M}) where {T <: Real, N, M, K, OpType<:BinaryOp{T}}
    val = forward(OpType, child1.val, child2.val);
    grad = zeros(T, size(val));
    return BinaryOpTensor{T, N, M, K, OpType}(val, grad, child1, child2);
end

############ PRODUCT

import Base: *

function forward(::Type{MultOp{T}}, X::Array{T, N}, Y::Array{T, M}) where {T<:Real, N, M}
    return Broadcast.broadcast(*, X, Y);
end

function backward!(tensor::BinaryOpTensor{T, N, M, K, MultOp{T}}) where {T<:Real, N, M, K}
    if !(tensor.child1 isa ConstantTensor)
        dx = similar(tensor.child1.val);
        sum!(dx, Broadcast.broadcast(*, tensor.child2.val, tensor.grad));
        update_gradient!(tensor.child1, dx);
    end

    if !(tensor.child2 isa ConstantTensor)
        dy = similar(tensor.child2.val);
        sum!(dy, Broadcast.broadcast(*, tensor.child1.val, tensor.grad));
        update_gradient!(tensor.child2, dy);
    end
end

*(X::Tensor{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = BinaryOpTensor{T, N, M, max(N, M), MultOp{T}}(X, Y)
*(X::Tensor{T, N}, Y::Array{T, M}) where{T<:Real, N, M} = X * ConstantTensor(Y)
*(X::Array{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = ConstantTensor(X) * Y
*(X::Tensor{T, N}, y::T) where{T<:Real, N} = X * [y]
*(x::T, Y::Tensor{T, M}) where{T<:Real, M} = [x] * Y

############ ADDITION

import Base: +

function forward(::Type{AddOp{T}}, X::Array{T, N}, Y::Array{T, M}) where {T<:Real, N, M}
    return Broadcast.broadcast(+, X, Y);
end

function backward!(tensor::BinaryOpTensor{T, N, M, K, AddOp{T}}) where {T<:Real, N, M, K}
    if !(tensor.child1 isa ConstantTensor)
        dx = similar(tensor.child1.val);
        sum!(dx, tensor.grad);
        update_gradient!(tensor.child1, dx);
    end

    if !(tensor.child2 isa ConstantTensor)
        dy = similar(tensor.child2.val);
        sum!(dy, tensor.grad);
        update_gradient!(tensor.child2, dy);
    end
end

+(X::Tensor{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = BinaryOpTensor{T, N, M, max(N, M), AddOp{T}}(X, Y)
+(X::Tensor{T, N}, Y::Array{T, M}) where{T<:Real, N, M} = X + ConstantTensor(Y)
+(X::Array{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = ConstantTensor(X) + Y
+(X::Tensor{T, N}, y::T) where{T<:Real, N} = X + [y]
+(x::T, Y::Tensor{T, M}) where{T<:Real, M} = [x] + Y

############ SUBSTRACTION

import Base: -

-(X::Tensor{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = BinaryOpTensor{T, N, M, max(N, M), AddOp{T}}(X, -Y)
-(X::Tensor{T, N}, Y::Array{T, M}) where{T<:Real, N, M} = X - ConstantTensor(Y)
-(X::Array{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = ConstantTensor(X) - Y
-(X::Tensor{T, N}, y::T) where{T<:Real, N} = X - [y]
-(x::T, Y::Tensor{T, M}) where{T<:Real, M} = [x] - Y


############ POW

import Base: ^

function forward(::Type{PowOp{T}}, X::Array{T, N}, Y::Array{T, M}) where {T<:Real, N, M}
    return Broadcast.broadcast(^, X, Y);
end

function backward!(tensor::BinaryOpTensor{T, N, M, K, PowOp{T}}) where {T<:Real, N, M, K}
    if !(tensor.child1 isa ConstantTensor)
        dx = similar(tensor.child1.val);
        sum!(dx, tensor.grad .* tensor.child2.val .* (tensor.child1.val .^ (tensor.child2.val .- 1.0)));
        update_gradient!(tensor.child1, dx);
    end

    if !(tensor.child2 isa ConstantTensor)
        dy = similar(tensor.child2.val);
        sum!(dy, tensor.grad .* log.(tensor.child2.val) .* (tensor.child1.val .^ tensor.child2.val));
        update_gradient!(tensor.child2, dy);
    end
end

^(X::Tensor{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = BinaryOpTensor{T, N, M, max(N, M), PowOp{T}}(X, Y)
^(X::Tensor{T, N}, Y::Array{T, M}) where{T<:Real, N, M} = X ^ ConstantTensor(Y)
^(X::Array{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = ConstantTensor(X) ^ Y
^(X::Tensor{T, N}, y::T) where{T<:Real, N} = X ^ [y]
^(x::T, Y::Tensor{T, M}) where{T<:Real, M} = [x] ^ Y


############ DIVISION

import Base: /

/(X::Tensor{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = BinaryOpTensor{T, N, M, max(N, M), MultOp{T}}(X, Y ^ T(-1.0))
/(X::Tensor{T, N}, Y::Array{T, M}) where{T<:Real, N, M} = X / ConstantTensor(Y)
/(X::Array{T, N}, Y::Tensor{T, M}) where{T<:Real, N, M} = ConstantTensor(X) / Y
/(X::Tensor{T, N}, y::T) where{T<:Real, N} = X / [y]
/(x::T, Y::Tensor{T, M}) where{T<:Real, M} = [x] / Y



