function layer = residual_block_constructor( layer , input, input_map_size )

l = 1;
block{l}.type = 'convolution';
block{l}.weight = layer.weight;
block{l}.weight.shape = [3, 3];
block{l}.bias.option = false;
block{l}.output = layer.output;
block{l}.zero_padding.option = true;
block{l} = convolution_constructor(block{1}, input, input_map_size);

if layer.sampling.option == true
    l = l + 1;
    layer.sampling.layer = l;
    block{l}.type = 'sampling';
    block{l}.sampling = layer.sampling;
    block{l} = sampling_constructor(block{l}, block{l-1}.output, block{l-1}.map_size);
end

l = l + 1;
block{l}.type = 'batch_normalization';
block{l}.BN_decay = layer.BN_decay;
block{l} = batch_normalization_constructor(block{l}, block{l-1}.output, block{l-1}.map_size);


l = l + 1;
block{l}.type = 'activation';
block{l}.activation = 'relu';
block{l} = activation_constructor(block{l}, block{l-1});

l = l + 1;
block{l}.type = 'convolution';
block{l}.weight = layer.weight;
block{l}.weight.shape = [3, 3];
block{l}.bias.option = false;
block{l}.output = layer.output;
block{l}.zero_padding.option = true;
block{l} = convolution_constructor(block{l}, block{l-1}.output, block{l-1}.map_size);

l = l + 1;
block{l}.type = 'residual';
block{l}.map_size = block{l-1}.map_size;
block{l}.output = block{l-1}.output;
block{l}.X = cell(block{l}.output, 1);

l = l + 1;
block{l}.type = 'batch_normalization';
block{l}.BN_decay = layer.BN_decay;
block{l} = batch_normalization_constructor(block{l}, block{l-1}.output, block{l-1}.map_size);

l = l + 1;
block{l}.type = 'activation';
block{l}.activation = 'relu';
block{l} = activation_constructor(block{l}, block{l-1});

layer.block = block;

layer.map_size  = block{l}.map_size;

layer.isTensor = true;

end
