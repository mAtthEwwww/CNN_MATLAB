function layer = SMP_feedforward( layer , X )

if strcmp(layer.sampling.method, 'max')

    for j = 1 : length(X)
        
        [layer.X{j}, layer.sampling.max_position{j}] = down_max_sampling(X{j}, layer.map_size, layer.sampling.shape, layer.sampling.stride, layer.zero_padding.width);

    end

elseif strcmp(layer.sampling.method, 'average')
    
    for j = 1 : length(X)
        
        layer.X{j} = down_averge_sampling(X{j}, layer.map_size, layer.sampling.shape, layer.sampling.stride, layer.zero_padding.width);
    
    end

elseif strcmp(layer.sampling.method, 'grid')

    for j = 1 : length(X)
        
        layer.X{j} = down_grid_sampling(X{j}, layer.map_size, layer.sampling.shape, layer.sampling.stride, layer.zero_padding.width);

    end

else

    error('sampling method wrong')

end

end
