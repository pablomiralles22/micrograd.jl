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

function show(io::IO, tensor::UnaryOpTensor{T, N, OpType}) where {T<:Real, N, OpType<:UnaryOp{T}}
    println("-----------------------------")
    println(io, "UnaryOpTensor with type ", T, ", operation ", OpType, " and size ", size(tensor.val))
    print(io, "\nValue: ")
    show(io, "text/plain", tensor.val)
    println()
    print(io, "\nGradient: ")
    show(io, "text/plain", tensor.grad)
    println("\n-----------------------------")
end


function show(io::IO, tensor::BinaryOpTensor{T, N, M, K, OpType}) where {T<:Real, N, M, K, OpType<:BinaryOp{T}}
    println("-----------------------------")
    println(io, "BinaryOpTensor with type ", T, ", operation ", OpType, " and size ", size(tensor.val))
    print(io, "\nValue: ")
    show(io, "text/plain", tensor.val)
    println()
    print(io, "\nGradient: ")
    show(io, "text/plain", tensor.grad)
    println("\n-----------------------------")
end

function show(io::IO, tensor::ContractionTensor{T, N, M, K, D}) where {T<:Real, N, M, K, D}
    println("-----------------------------")
    println(io, "ContractionTensor with type ", T, ", dimension ", D, " and size ", size(tensor.val))
    print(io, "\nValue: ")
    show(io, "text/plain", tensor.val)
    println()
    print(io, "\nGradient: ")
    show(io, "text/plain", tensor.grad)
    println("\n-----------------------------")
end

