function pre_delta = BP_activation_layer( layer , pre_X )

derived_activation = str2func(['derived_', layer.activation]);

if layer.isTensor == true
    
    pre_delta = cell(length(pre_X), 1);

    for i = 1 : length(pre_X)

        pre_delta{i} = derived_activation(pre_X{i}) .* layer.delta{i};

    end

else

    pre_delta = derived_activation(pre_X) .* layer.delta;

end

end
