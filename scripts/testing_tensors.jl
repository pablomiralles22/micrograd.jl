using micrograd

a = DifferentiableTensor(randn(Float32, 3, 3) .* 10)
b = Float32(2.0) / a
c = exp(b)

c.grad = ones(size(c.grad))
backpropagate!(c)

for tensor in build_topological_sort(c)
    show(tensor)
end
