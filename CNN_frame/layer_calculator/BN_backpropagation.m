function [layer, pre_delta] = BN_backpropagation( layer )

pre_delta = cell(length(layer.delta), 1);

for j = 1 : length(layer.delta)
    tmp = layer.delta{j} .* layer.X_normalized{j};
    layer.gamma.grad{j} = sum(tmp(:));
    layer.beta.grad{j} = sum(layer.delta{j}(:));

    layer.delta{j} = layer.gamma.g{j} * layer.delta{j};
    
    tmp = layer.delta{j} .* (layer.X{j} - layer.mu.mini_batch{j});
    delta_sigma2 = sum(tmp(:)) * (-1 / 2) * layer.sigma2.mini_batch{j}^(-3 / 2);

    tmp = layer.X{j} - layer.mu.mini_batch{j};   
    delta_mu = sum(layer.delta{j}(:)) * (-1) / sqrt(layer.sigma2.mini_batch{j}) + delta_sigma2 * (-2) * sum(tmp(:)) / prod(size(layer.X{j}));

    pre_delta{j} = layer.delta{j} / sqrt(layer.sigma2.mini_batch{j}) + delta_sigma2 * 2 * tmp / prod(size(layer.X{j})) + delta_mu / prod(size(layer.X{j}));
end

end
