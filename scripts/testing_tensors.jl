using micrograd

a = BaseDifferentiableTensor(randn(Float32, 3, 3) .* 10)

b = tensor_tanh(a)
b.grad[1,1] = 1
backward(b)

println(a.val)
println(a.grad)
println()
println(b.val)
println(b.grad)