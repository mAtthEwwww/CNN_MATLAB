function layer = sampling_constructor( layer , input , input_map_size )

layer.vector = false;

layer.output = input;

tmp = ceil( ( input_map_size - layer.sampling_size ) / layer.stride );

layer.pad_size = tmp * layer.stride + layer.sampling_size - input_map_size;

layer.map_size = tmp + 1;

end