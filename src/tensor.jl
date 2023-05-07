export Tensor, ConstantTensor, DifferentiableTensor, DifferentiableTensor,
    update_gradient, update_gradient, backward

#############
############# TENSOR CLASSES
#############

abstract type Tensor{T<:Real, N} end

struct ConstantTensor{T<:Real, N} <: Tensor{T, N}
    val::Array{T, N} 
end

abstract type BaseDifferentiableTensor{T<:Real, N} <: Tensor{T, N} end

mutable struct DifferentiableTensor{T<:Real, N} <: BaseDifferentiableTensor{T, N}
    val:: Array{T, N}
    grad::Array{T, N}
end

DifferentiableTensor(val::Array{T, N}) where {T, N} = DifferentiableTensor(val::Array{T, N}, zeros(T, size(val)))

#############
############# UPDATE GRADIENT & RESET GRADIENT
#############

function update_gradient(::ConstantTensor{T, N}, ::Array{T, N}) where {T, N}
end

function update_gradient(tensor::BaseDifferentiableTensor{T, N}, delta::Array{T, N}) where {T, N}
    tensor.grad += delta
end

function reset_gradient(tensor::ConstantTensor{T, N}) where {T, N}
end

function reset_gradient(tensor::BaseDifferentiableTensor{T, N}) where {T, N}
    fill!(tensor.grad, 0);
end

#############
############# BACKWARD PASS
#############

function backward(tensor::DifferentiableTensor)
end