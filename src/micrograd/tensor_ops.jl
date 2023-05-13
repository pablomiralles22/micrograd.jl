using LoopVectorization

############ TENSOR CONTRACTION ############ 

mutable struct ContractionTensor{T<:Real, N, M, K, D} <: BaseDifferentiableTensor{T, K}
    val::Array{T, K}
    grad::Array{T, K}
    child1::Tensor{T, N}
    child2::Tensor{T, M}

    function ContractionTensor(child1::Tensor{T, N}, child2::Tensor{T, M}, D::Integer) where {T<:Real, N, M}
        @assert N isa Integer
        @assert M isa Integer
        @assert D isa Integer
        @assert N >= D
        @assert M >= D

        K::Integer = max(N + M - 2*D, 1)

        sz1_fix = size(child1.val)[1:(N-D)]
        sz1_contract = size(child1.val)[(N-D+1):N]
        sz2_fix = size(child2.val)[1:(M-D)]
        sz2_contract = size(child2.val)[(M-D+1):M]

        @assert sz1_contract == sz2_contract

        sz = N + M - 2*D > 0 ? (sz1_fix..., sz2_fix...) : (1)

        val = zeros(sz)
        grad = zeros(sz)

        contract_indices = CartesianIndices(sz1_contract)
        @tturbo for ind1 in CartesianIndices(sz1_fix), ind2 in CartesianIndices(sz2_fix), ind_contract in contract_indices
            val[ind1, ind2] += child1.val[ind1, ind_contract] * child2.val[ind2, ind_contract]
        end

        new{T, N, M, K, D}(val, grad, child1, child2)
    end
end

function tensor_contraction(t1::Tensor{T, N}, t2::Tensor{T, M}, D::Integer) where {T<:Real, N, M}
    return ContractionTensor(t1, t2, D)
end

export tensor_contraction

function backward!(tensor::ContractionTensor{T, N, M, K, D}) where {T<:Real, N, M, K, D}
    if !(tensor.child1 isa ConstantTensor)
        dx = similar(tensor.child1.val);
        sum!(dx, tensor.grad .* tensor.child2.val .* (tensor.child1.val .^ (tensor.child2.val .- 1.0)));
        update_gradient!(tensor.child1, dx);
    end
    #
    if !(tensor.child2 isa ConstantTensor)
        dy = similar(tensor.child2.val);
        sum!(dy, tensor.grad .* log.(tensor.child2.val) .* (tensor.child1.val .^ tensor.child2.val));
        update_gradient!(tensor.child2, dy);
    end
end

