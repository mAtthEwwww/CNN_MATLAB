function layer = sampling_constructor( layer , input , input_map_size )

layer.isTensor = true;

layer.output = input;

layer.X = cell(input, 1);

if isfield(layer.sampling, 'shape') == false
  
    layer.sampling.shape = layer.sampling.stride

elseif isfield(layer.sampling, 'stride') == false

    layer.sampling.stride = layer.sampling.shape;

end


tmp = ceil((input_map_size - layer.sampling.shape) ./ layer.sampling.stride);

layer.zero_padding.width = tmp .* layer.sampling.stride + layer.sampling.shape - input_map_size;

layer.map_size = tmp + 1;

end
