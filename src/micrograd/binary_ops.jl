abstract type BinaryOp{T<:Real} end

abstract type MultOp{T<:Real} <: BinaryOp{T} end
abstract type AddOp{T<:Real} <: BinaryOp{T} end