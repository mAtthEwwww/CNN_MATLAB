function layer = batch_normalization_constructor( layer, input, input_map_size )

% batch normalization layer after convolution layer

layer.isTensor = true;

layer.output = input;

layer.map_size = input_map_size;

layer.first_train = true;

layer.X = cell(layer.output, 1);

layer.gamma.g = cell(layer.output, 1);
layer.gamma.grad = cell(layer.output, 1);
layer.gamma.momentum = cell(layer.output, 1);
layer.beta.b = cell(layer.output, 1);
layer.beta.grad = cell(layer.output, 1);
layer.beta.momentum = cell(layer.output, 1);
layer.mu.mini_batch = cell(layer.output, 1);
layer.mu.moving_average = cell(layer.output, 1);

layer.sigma2.mini_batch = cell(layer.output, 1);
layer.sigma2.moving_average = cell(layer.output, 1);

for j = 1 : layer.output
    layer.gamma.g{j} = 1;
    layer.gamma.grad{j} = 0;
    layer.gamma.momentum{j} = 0;

    layer.beta.b{j} = 0;
    layer.beta.grad{j} = 0;
    layer.beta.momentum{j} = 0;

    layer.mu.moving_average{j} = 0;
    layer.sigma2.moving_average{j} = 0;
end

end
