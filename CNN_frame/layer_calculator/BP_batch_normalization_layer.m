function [layer, pre_delta] = BP_batch_normalization_layer( layer )

pre_delta = cell(length(layer.delta), 1);

for j = 1 : length(layer.delta)
    delta_X_normalized{j} = layer.gamma.g{j} * layer.delta{j};
    
    tmp = delta_X_normalized{j} .* (layer.X{j} - layer.mu.mini_batch{j});
    delta_sigma2 = sum(tmp(:)) * (-1 / 2) * layer.sigma2.mini_batch{j}^(-3 / 2);

    tmp = layer.X{j} - layer.mu.mini_batch{j};   
    delta_mu = sum(delta_X_normalized{j}(:)) * (-1) / sqrt(layer.sigma2.mini_batch{j}) + delta_sigma2 * (-2) * sum(tmp(:)) / prod(size(layer.X{j}));

    pre_delta{j} = delta_X_normalized{j} / sqrt(layer.sigma2.mini_batch{j}) + delta_sigma2 * 2 * tmp / prod(size(layer.X{j})) + delta_mu / prod(size(layer.X{j}));
    %layer.gamma.g{j}
    %layer.beta.b{j}
    %sum(pre_delta{j}(:))

    tmp = layer.delta{j} .* layer.X_normalized{j};
    layer.gamma.grad{j} = sum(tmp(:));
    layer.beta.grad{j} = sum(layer.delta{j}(:));
end

end
