export Tensor, ConstantTensor, DifferentiableTensor, DifferentiableTensor,
    update_gradient!, update_gradient!, backward!

#############
############# BASE TENSOR CLASSES
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
# TODO remove

function update_gradient!(::ConstantTensor{T, N}, ::Array{T, N}) where {T, N}
end

function update_gradient!(tensor::BaseDifferentiableTensor{T, N}, delta::Array{T, N}) where {T, N}
    tensor.grad += delta
end

function reset_gradient!(tensor::ConstantTensor{T, N}) where {T, N}
end

function reset_gradient!(tensor::BaseDifferentiableTensor{T, N}) where {T, N}
    fill!(tensor.grad, 0);
end

#############
############# backward pass
#############

function backward!(tensor::ConstantTensor)
end

function backward!(tensor::DifferentiableTensor)
end

#############
############# PRINTING
#############
import Base: show
function show(io::IO, tensor::ConstantTensor{T, N}) where {T, N}
    println("-----------------------------")
    println(io, "ConstantTensor with type ", T, " and size ", size(tensor.val))
    print(io, "\nValue: ")
    show(io, "text/plain", tensor.val)
    println("\n-----------------------------")
end

function show(io::IO, tensor::DifferentiableTensor{T, N}) where {T, N}
    println("-----------------------------")
    println(io, "DifferentiableTensor with type ", T, " and size ", size(tensor.val))
    print(io, "\nValue: ")
    show(io, "text/plain", tensor.val)
    println()
    print(io, "\nGradient: ")
    show(io, "text/plain", tensor.grad)
    println("\n-----------------------------")
end