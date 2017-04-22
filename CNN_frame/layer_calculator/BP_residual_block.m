function [block, pre_delta] = BP_residual_layer( block , pre_X )

block.layer{length(block.layer)}.delta = block.delta;

for l = length(block.layer) : -1 : 2
    if strcmp(block.layer{l}.type, 'convolution')
        [block.layer{l}, block.layer{l-1}.delta] = BP_convolution_layer(block.layer{l}, block.layer{l-1}.X);

    elseif strcmp(block.layer{l}.type, 'batch_normalization')
        [block.layer{l}, block.layer{l-1}.delta] = BP_batch_normalization_layer(block.layer{l});

    elseif strcmp(block.layer{l}.type, 'sampling')
        block.layer{l-1}.delta = BP_sampling_layer(block.layer{l}, block.layer{l-1}.map_size);

    elseif strcmp(block.layer{l}.type, 'activation')
        block.layer{l-1}.delta = BP_activation_layer(block.layer{l}, block.layer{l-1}.X);

    elseif strcmp(block.layer{l}.type, 'short_cut')
        block.layer{l-1}.delta = block.layer{l}.delta;
        if block.sampling.option == true
            shortcut_samp = block.layer{block.sampling.block};
            shortcut_samp.delta = block.layer{l}.delta;
            shortcut_delta = BP_sampling_layer(shortcut_samp, size(pre_X{1}(:,:,1)));
        else
            shortcut_delta = block.layer{l}.delta;
        end
    else
        error('layer type wrong')
    end
end

[block.layer{1}, pre_delta] = BP_convolution_layer(block.layer{1}, pre_X);

for i = 1 : length(pre_delta)
    
    pre_delta{i} = pre_delta{i} + shortcut_delta{i};

end

end
