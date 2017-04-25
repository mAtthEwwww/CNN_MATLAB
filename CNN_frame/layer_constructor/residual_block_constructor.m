function block = residual_block_constructor( block , input, input_map_size )

l = 1;
layer{l}.type = 'convolution';
layer{l}.weight = block.weight;
layer{l}.weight.shape = [3, 3];
layer{l}.bias.option = false;
layer{l}.output = block.output;
layer{l}.zero_padding.option = true;
layer{l} = convolution_constructor(layer{l}, input, input_map_size);

if block.sampling.option == true
    l = l + 1;
    block.sampling.layer = l;
    layer{l}.type = 'sampling';
    layer{l}.sampling = block.sampling;
    layer{l} = sampling_constructor(layer{l}, layer{l-1}.output, layer{l-1}.map_size);
end

l = l + 1;
layer{l}.type = 'batch_normalization';
layer{l}.BN_decay = block.BN_decay;
layer{l} = batch_normalization_constructor(layer{l}, layer{l-1}.output, layer{l-1}.map_size);


l = l + 1;
layer{l}.type = 'activation';
layer{l}.activation = 'relu';
layer{l} = activation_constructor(layer{l}, layer{l-1});

l = l + 1;
layer{l}.type = 'convolution';
layer{l}.weight = block.weight;
layer{l}.weight.shape = [3, 3];
layer{l}.bias.option = false;
layer{l}.output = block.output;
layer{l}.zero_padding.option = true;
layer{l} = convolution_constructor(layer{l}, layer{l-1}.output, layer{l-1}.map_size);

l = l + 1;
layer{l}.type = 'short_cut';
layer{l}.map_size = layer{l-1}.map_size;
layer{l}.output = layer{l-1}.output;
if block.sampling.option == true
    layer{l}.smp.sampling = block.sampling;
    layer{l}.smp = sampling_constructor(layer{l}.smp, input, input_map_size);
    layer{l}.map_size = layer{l}.smp.map_size;
end
if layer{l}.output ~= input
    layer{l}.conv.weight = block.weight;
    layer{l}.conv.weight.shape = [1, 1];
    layer{l}.conv.bias.option = false;
    layer{l}.conv.zero_padding.option = false;
    layer{l}.conv.output = layer{l-1}.output;
    layer{l}.conv = convolution_constructor(layer{l}.conv, input, layer{l}.map_size);
    layer{l}.bn.BN_decay = block.BN_decay;
    layer{l}.bn = batch_normalization_constructor(layer{l}.bn, layer{l}.output, layer{l}.map_size);
end
%layer{l}.X = cell(layer{l}.output, 1);

l = l + 1;
layer{l}.type = 'batch_normalization';
layer{l}.BN_decay = block.BN_decay;
layer{l} = batch_normalization_constructor(layer{l}, layer{l-1}.output, layer{l-1}.map_size);

l = l + 1;
layer{l}.type = 'activation';
layer{l}.activation = 'relu';
layer{l} = activation_constructor(layer{l}, layer{l-1});

block.layer = layer;

block.map_size  = layer{l}.map_size;

block.isTensor = true;

end
