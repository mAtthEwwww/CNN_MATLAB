function layer = full_connection_constructor( layer , layer_input )

layer.isTensor = false;

if layer_input.isTensor == true
    fan_in = layer_input.output * prod(layer_input.map_size);
else
    fan_in = layer_input.output;
end

shape = [layer.output, fan_in];
layer.weight.W = filler(fan_in, layer.output, shape, layer.weight.filler);
layer.weight.grad = zeros(shape);
layer.weight.momentum = zeros(shape);

layer.bias.b = zeros(1, layer.output);
layer.bias.grad = zeros(1, layer.output);
layer.bias.momentum = zeros(1, layer.output);

end
