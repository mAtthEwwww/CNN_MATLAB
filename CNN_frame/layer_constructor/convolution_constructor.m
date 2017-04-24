function layer = convolution_constructor( layer , input, input_map_size )

if layer.zero_padding.option == true
    layer.map_size = input_map_size;
    layer.zero_padding.width = (layer.weight.shape - 1) / 2;
else
    layer.map_size = input_map_size - layer.weight.shape + 1;
    layer.zero_padding.width = [0, 0];
end

layer.zero_padding.inv_width = layer.weight.shape - 1 - layer.zero_padding.width;

layer.isTensor = true;

layer.X = cell(layer.output, 1);

layer.weight.kernel = cell(layer.output, input);
layer.weight.grad = cell(layer.output, input);
layer.weight.momentum = cell(layer.output, input);

for j = 1 : layer.output
    for i = 1 : input
        layer.weight.kernel{j,i} = filler(input, layer.output, layer.weight.shape, layer.weight.filler);
        layer.weight.grad{j,i} = zeros(layer.weight.shape);
        layer.weight.momentum{j,i} = zeros(layer.weight.shape);
    end

end

if layer.bias.option == true

    layer.bias.b = cell(layer.output, 1);
    layer.bias.grad = cell(layer.output, 1);
    layer.bias.momentum = cell(layer.output, 1);

    for j = 1 : layer.output
        layer.bias.b{j} = 0;
        layer.bias.grad{j} = 0;
        layer.bias.momentum{j} = 0;
    end
end

end
