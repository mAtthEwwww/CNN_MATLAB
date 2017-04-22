function layer = FF_sampling_layer( layer , X )

if strcmp(layer.sampling.type, 'max')

    for j = 1 : length(X)
        
        [layer.X{j}, layer.sampling.max_position{j}] = down_max_sampling(X{j}, layer.map_size, layer.sampling.shape, layer.sampling.stride, layer.zero_padding.width);

    end

elseif strcmp(layer.sampling.type, 'average')
    
    for j = 1 : length(X)
        
        layer.X{j} = down_average_sampling(X{j}, layer.map_size, layer.sampling.shape, layer.sampling.stride, layer.zero_padding.width);
    
    end

elseif strcmp(layer.sampling.type, 'grid')

    for j = 1 : length(X)
        
        layer.X{j} = down_grid_sampling(X{j}, layer.map_size, layer.sampling.shape, layer.sampling.stride, layer.zero_padding.width);

    end

else

    error('sampling type wrong')

end

end
