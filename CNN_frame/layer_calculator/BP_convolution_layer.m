function [layer, pre_delta] = BP_convolution_layer( layer, pre_X )

%[~, ~, N] = size(pre_X{1});

for j = 1 : size(layer.weight.kernel, 1)

    for i = 1 : size(layer.weight.kernel, 2)

        layer.weight.grad{j,i} = convolution(rot90(pre_X{i}, 2), layer.delta{j}(:, :, end:-1:1), layer.zero_padding.width);

    end

end

if layer.bias.option == true
    
    for j = 1 : size(layer.weight.kernel, 1)
        
        layer.bias.grad{j} = sum(layer.delta{j}(:));

    end

end


% if layer contains key word bottom
if isfield(layer, 'bottom') == true

    pre_delta = {};

else
    
    pre_delta = cell(size(layer.weight.kernel, 2), 1);

    for i = 1 : size(layer.weight.kernel, 2)
    
        pre_delta{i} = zeros(size(pre_X{1}));

        for j = 1 : size(layer.weight.kernel, 1)

            pre_delta{i} = pre_delta{i} + convolution(layer.delta{j}, rot90(layer.weight.kernel{j,i}, 2), layer.zero_padding.inv_width);

        end

    end

end

end
