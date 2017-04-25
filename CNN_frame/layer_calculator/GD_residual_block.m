function block = GD_residual_block( block , learning_rate , momentum , weight_decay )

for l = 1 : length(block.layer)
    if strcmp(block.layer{l}.type, 'convolution')
        block.layer{l} = GD_convolution_layer(block.layer{l}, learning_rate, momentum, weight_decay);

    elseif strcmp(block.layer{l}.type, 'batch_normalization')
        block.layer{l} = GD_batch_normalization_layer(block.layer{l}, learning_rate, momentum);

    elseif strcmp(block.layer{l}.type, 'full_connection')
        block.layer{l} = GD_batch_normalization_layer(block.layer{l}, learning_rate, momentum, weight_decay);

    elseif strcmp(block.layer{l}.type, 'short_cut')
        if isfield(block.layer{l}, 'conv')
            block.layer{l}.bn = GD_batch_normalization_layer(block.layer{l}.bn, learning_rate, momentum);
            block.layer{l}.conv = GD_convolution_layer(block.layer{l}.conv, learning_rate, momentum, weight_decay);
        end

    elseif strcmp(block.layer{l}.type, 'sampling')
        ;

    elseif strcmp(block.layer{l}.type, 'activation')
        ;

    else
        error('layer type wrong')

    end
end

end
