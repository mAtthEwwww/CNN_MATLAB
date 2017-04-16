function pre_delta = SMP_backpropagation( layer, pre_map_size )

pre_delta = cell(length(layer.delta), 1);

if strcmp( layer.sampling.method, 'max' )

    for j = 1 : length(layer.delta)

        pre_delta{j} = up_max_sampling( layer.delta{j}, layer.sampling.max_position{j}, pre_map_size, layer.sampling.shape, layer.sampling.stride , layer.zero_padding.width);

    end

elseif strcmp( layer.sampling.method, 'average' )

    for j = 1 : length(layer.delta)

        pre_delta{j} = up_average_sampling( layer.delta{j} , pre_map_size , layer.sampling.shape, layer.sampling.stride , layer.zero_padding.width);

    end

elseif strcmp(layer.sampling.method, 'grid')
    
    for j = 1 : length(layer.delta)
        
        pre_delta{j} = up_grid_sampling(layer.delta{j}, pre_map_size, layer.sampling.shape, layer.sampling.stride, layer.zero_padding.width);

    end

else

    error('sampling method wrong')

end

end
