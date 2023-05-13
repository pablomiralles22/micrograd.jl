using micrograd

a = DifferentiableTensor(randn(Float32, 3, 3) .* 10)

b = Float32(2.0) / a
b.grad = ones(3,3)
backward!(b)
backward!(b.child2)

println(a.val)
println(a.grad)
println(- Float32(2.0) ./ (a.val .^ Float32(2)))
println()
println(b.val)
println(b.grad)
