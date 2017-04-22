function layer = FF_full_connection_layer( layer , X , pre_isTensor, isTrain )

if pre_isTensor == true

    [R, C, N] = size(X{1});

    layer.pre_X = zeros(N, R * C * length(X));

    for i = 1 : length(X)
        
        layer.pre_X(:, (i - 1) * R * C + 1 : i * R * C) = reshape(X{i}, R * C, N)';

    end

    layer.X = bsxfun(@plus, layer.pre_X * layer.weight.W', layer.bias.b);

else
    
    layer.X = bsxfun(@plus, X * layer.weight.W', layer.bias.b);

end


if layer.dropout.option == true
    
    if isTrain == true
        
        layer.dropout.mask = binornd(1, 1 - layer.dropout.rate, size(layer.X));

        layer.X = layer.X .* layer.dropout.mask;

    else
        
        layer.X = layer.X * (1 - layer.dropout.rate);

    end

end

end
