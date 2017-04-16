function [layer, pre_delta] = FC_backpropagation( layer , pre_X , pre_isTensor ) 

if layer.dropout.option == true
    
    layer.delta = layer.delta .* layer.dropout.mask;

end


if pre_isTensor == true
    
    layer.weight.grad = layer.delta' * layer.pre_X;

else
    
    layer.weight.grad = layer.delta' * pre_X;

end

layer.bias.grad = sum(layer.delta, 1);


% if layer contains key word bottom
if isfield(layer, 'bottom') == true

    pre_delta = {};

else

    if pre_isTensor == true

        pre_delta_tmp = layer.delta * layer.weight.W;
            
        [R, C, N] = size(pre_X{1});

        pre_delta = cell(length(pre_X), 1);

        for i = 1 : length(pre_X)

            pre_delta{i} = reshape(pre_delta_tmp(:, (i-1)*R*C+1 : i*R*C)', R, C, N);

        end

    else

        pre_delta = layer.delta * layer.weight.W;

    end

end

end
