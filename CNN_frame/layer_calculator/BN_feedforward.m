function layer = BN_feedforward( layer, X, isTrain )
% batch normalization feedforward function

layer.X_normalized = cell(length(X), 1);

if isTrain == true
    for j = 1 : layer.output
        layer.mu.mini_batch{j} = mean(X{j}(:));
        layer.sigma2.mini_batch{j} = var(X{j}(:), 1);

        layer.mu.moving_average{j} = layer.BN_decay * layer.mu.moving_average{j} + (1 - layer.BN_decay) * layer.mu.mini_batch{j};
        layer.sigma2.moving_average{j} = layer.BN_decay * layer.sigma2.moving_average{j} + (1 - layer.BN_decay) * layer.sigma2.mini_batch{j};

        layer.X_normalized{j} = (X{j} - layer.mu.mini_batch{j}) / sqrt(layer.sigma2.mini_batch{j});
        layer.X{j} = layer.gamma.g{j} * layer.X_normalized{j} + layer.beta.b{j};
    end
else
    for j = 1 : layer.output
        layer.X_normalized{j} = (X{j} - layer.mu.moving_average{j}) / sqrt(layer.sigma2.moving_average{j});
        layer.X{j} = layer.gamma.g{j} * layer.X_normalized{j} + layer.beta.b{j};
    end
end

end
