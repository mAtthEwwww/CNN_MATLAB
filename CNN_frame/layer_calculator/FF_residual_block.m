function block = FF_residual_block( block , X , isTrain )


[~, ~, N] = size(X{1});

block.layer{1} = FF_convolution_layer(block.layer{1}, X);

for l = 2 : length(block.layer)
    if strcmp(block.layer{l}.type, 'convolution')
        block.layer{l} = FF_convolution_layer(block.layer{l}, block.layer{l-1}.X);

    elseif strcmp(block.layer{l}.type, 'batch_normalization')
        block.layer{l} = FF_batch_normalization_layer(block.layer{l}, block.layer{l-1}.X, isTrain);

    elseif strcmp(block.layer{l}.type, 'sampling')
        block.layer{l} = FF_sampling_layer(block.layer{l}, block.layer{l-1}.X);

    elseif strcmp(block.layer{l}.type, 'activation')
        block.layer{l} = FF_activation_layer(block.layer{l}, block.layer{l-1}.X);

    elseif strcmp(block.layer{l}.type, 'short_cut')
        block.layer{l}.X = block.layer{l-1}.X;
        if isfield(block.layer{l}, 'smp')
            block.layer{l}.smp = FF_sampling_layer(block.layer{l}.smp, X);
            X = block.layer{l}.smp.X;
        end
        if isfield(block.layer{l}, 'conv')
            block.layer{l}.conv.pre_X = X;
            block.layer{l}.conv = FF_convolution_layer(block.layer{l}.conv, block.layer{l}.conv.pre_X);
            block.layer{l}.bn = FF_batch_normalization_layer(block.layer{l}.bn, block.layer{l}.conv.X, isTrain);
            X = block.layer{l}.bn.X;
        end
        for i = 1 : length(X)
            block.layer{l}.X{i} = block.layer{l}.X{i} + X{i};
        end

    else
        error('layer type wrong')
    end

end

block.X = block.layer{length(block.layer)}.X;

end
