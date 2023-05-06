export Tensor, ConstantTensor, DifferentiableTensor, BaseDifferentiableTensor,
    update_gradient, update_gradient, backward

#############
############# TENSOR CLASSES
#############

abstract type Tensor{T<:Real, N} end

struct ConstantTensor{T<:Real, N} <: Tensor{T, N}
    val::Array{T, N} 
end

abstract type DifferentiableTensor{T<:Real, N} <: Tensor{T, N} end

mutable struct BaseDifferentiableTensor{T<:Real, N} <: DifferentiableTensor{T, N}
    val:: Array{T, N}
    grad::Array{T, N}
end

BaseDifferentiableTensor(val::Array{T, N}) where {T, N} = BaseDifferentiableTensor(val::Array{T, N}, zeros(T, size(val)))

# UNARY OPS

mutable struct ExpDifferentiableTensor{T<:Real, N} <: DifferentiableTensor{T, N}
    val:: Array{T, N}
    grad::Array{T, N}
    child::Tensor{T, N}
end

#############
############# UPDATE GRADIENT & RESET GRADIENT
#############

function update_gradient(tensor::ConstantTensor{T, N}, delta::Array{T, N}) where {T, N}
end

function update_gradient(tensor::DifferentiableTensor{T, N}, delta::Array{T, N}) where {T, N}
    tensor.grad += delta
end

function reset_gradient(tensor::ConstantTensor{T, N}) where {T, N}
end

function reset_gradient(tensor::DifferentiableTensor{T, N}) where {T, N}
    fill!(tensor.grad, 0);
end

#############
############# BACKWARD PASS
#############

function backward(tensor::BaseDifferentiableTensor)
end