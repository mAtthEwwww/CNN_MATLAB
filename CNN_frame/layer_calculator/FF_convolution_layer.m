function layer = FF_convolution_layer( layer , X )

[~, ~, N] = size(X{1});

for j = 1 : size(layer.weight.kernel, 1)

    layer.X{j} = zeros([layer.map_size, N]);

    for i = 1 : size(layer.weight.kernel, 2)

        layer.X{j} = layer.X{j} + convolution(X{i}, layer.weight.kernel{j,i}, layer.zero_padding.width);

    end

end

if layer.bias.option == true

    for j = 1 : layer.output

        layer.X{j} = layer.X{j} + layer.bias.b{j};

    end

end

end
