using DataStructures

function get_children(tensor::ConstantTensor)::Vector{Tensor}
    return []
end

function get_children(tensor::DifferentiableTensor)::Vector{Tensor}
    return []
end

function get_children(tensor::UnaryOpTensor)::Vector{Tensor}
    return [tensor.child]
end

function get_children(tensor::BinaryOpTensor)::Vector{Tensor}
    return [tensor.child1, tensor.child2]
end

function build_topological_sort(tensor::Tensor)::Vector{Tensor}
    res::Vector{Tensor} = [];

    visited = IdDict{Tensor, Bool}();
    is_visited(t::Tensor)::Bool = haskey(visited, t);
    mark_as_visited(t::Tensor) = visited[t] = true;

    stack = Stack{Tensor}();
    push!(stack, tensor);

    while length(stack) > 0
        node = first(stack)
        children_to_vist = [child for child in get_children(node) if !is_visited(child)]

        if length(children_to_vist) == 0
            pop!(stack)
            push!(res, node)
            mark_as_visited(node)
        else
            for child in children_to_vist
                push!(stack, child)
            end
        end
    end

    return reverse(res)
end

function backpropagate!(tensor::Tensor)
    for t in build_topological_sort(tensor)
        backward!(t)
    end
end


export backpropagate!, build_topological_sort