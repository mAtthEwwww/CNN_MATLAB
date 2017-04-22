function block = GD_residual_block( block , config )

for l = 1 : length(block.layer)
    if strcmp(block.layer{l}.type, 'convolution')
        block.layer{l} = GD_convolution_layer(block.layer{l}, config);

    elseif strcmp(block.layer{l}.type, 'batch_normalization')
        block.layer{l} = GD_batch_normalization_layer(block.layer{l}, config);

    elseif strcmp(block.layer{l}.type, 'full_connection')
        block.layer{l} = GD_batch_normalization_layer(block.layer{l}, config);

    end
end

end
