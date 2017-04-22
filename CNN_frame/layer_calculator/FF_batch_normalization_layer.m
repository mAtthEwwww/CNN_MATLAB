function layer = FF_batch_normalization_layer( layer, X, isTrain )
% batch normalization feedforward function

layer.X_normalized = cell(length(X), 1);

if isTrain == true
    for j = 1 : layer.output
        layer.mu.mini_batch{j} = mean(X{j}(:));
        layer.sigma2.mini_batch{j} = var(X{j}(:), 1);

        layer.X_normalized{j} = (X{j} - layer.mu.mini_batch{j}) / sqrt(layer.sigma2.mini_batch{j});
        layer.X{j} = layer.gamma.g{j} * layer.X_normalized{j} + layer.beta.b{j};
    end

    if layer.first_train == true
        layer.first_train = false;
        for j = 1 : layer.output
            layer.mu.moving_average{j} = layer.mu.mini_batch{j};
            layer.sigma2.moving_average{j} = layer.sigma2.mini_batch{j};
        end
    else
        for j = 1 : layer.output
            layer.mu.moving_average{j} = layer.BN_decay * layer.mu.moving_average{j} + (1 - layer.BN_decay) * layer.mu.mini_batch{j};
            layer.sigma2.moving_average{j} = layer.BN_decay * layer.sigma2.moving_average{j} + (1 - layer.BN_decay) * layer.sigma2.mini_batch{j};
        end
    end

else
    for j = 1 : layer.output
        layer.X_normalized{j} = (X{j} - layer.mu.moving_average{j}) / sqrt(layer.sigma2.moving_average{j});
        layer.X{j} = layer.gamma.g{j} * layer.X_normalized{j} + layer.beta.b{j};
    end
end

end
