function layer = FF_activation_layer( layer , X )

activation = str2func(layer.activation);

if layer.isTensor == true
    
    for j = 1 : length(X)
        
        layer.X{j} = activation(X{j});

    end

else

    layer.X = activation(X);

end

end
