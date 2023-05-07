
include("unary_ops.jl")

mutable struct UnaryOpTensor{T<:Real, N, OpType<:UnaryOp{T}} <: BaseDifferentiableTensor{T, N}
    val:: Array{T, N}
    grad::Array{T, N}
    child::Tensor{T, N}
end

function backward!(tensor::UnaryOpTensor{T, N, OpType}) where {T, N, OpType}
    delta = tensor.grad .* dop_apply(OpType, tensor.child.val);
    update_gradient!(tensor.child, delta);
end

function tensor_op(::Type{OpType}, tensor::Tensor{T, N})::UnaryOpTensor{T, N, OpType} where {T, N, OpType}
    val = op_apply(OpType, tensor.val);
    grad = zeros(T, size(val));
    return UnaryOpTensor{T, N, OpType}(val, grad, tensor)
end



import Base: exp, tanh, log, -

exp(tensor::Tensor{T, N}) where {T, N} = tensor_op(ExpOp{T}, tensor)
relu(tensor::Tensor{T, N}) where {T, N} = tensor_op(ReluOp{T}, tensor)
tanh(tensor::Tensor{T, N}) where {T, N} = tensor_op(TanhOp{T}, tensor)
log(tensor::Tensor{T, N}) where {T, N} = tensor_op(LogOp{T}, tensor)
-(tensor::Tensor{T, N}) where {T, N} = tensor_op(UnaryMinusOp{T}, tensor)

export exp, relu, tanh, log, -