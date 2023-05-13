module micrograd

include("micrograd/tensor.jl")
include("micrograd/unary_tensor_ops.jl")
include("micrograd/binary_tensor_ops.jl")
include("micrograd/tensor_ops.jl")

include("micrograd/tensor_graph.jl")
include("micrograd/tensor_printing.jl")

end # module micrograd
