function [layer, pre_delta] = RESBLK_backpropagation( layer , pre_X )

layer.block{length(layer.block)}.delta = layer.delta;

for l = length(layer.block) : -1 : 2
    if strcmp(layer.block{l}.type, 'convolution')
        [layer.block{l}, layer.block{l-1}.delta] = CN_backpropagation(layer.block{l}, layer.block{l-1}.X);

    elseif strcmp(layer.block{l}.type, 'batch_normalization')
        [layer.block{l}, layer.block{l-1}.delta] = BN_backpropagation(layer.block{l});

    elseif strcmp(layer.block{l}.type, 'sampling')
        layer.block{l-1}.delta = SMP_backpropagation(layer.block{l}, layer.block{l-1}.map_size);

    elseif strcmp(layer.block{l}.type, 'activation')
        layer.block{l-1}.delta = ACT_backpropagation(layer.block{l}, layer.block{l-1}.X);

    elseif strcmp(layer.block{l}.type, 'residual')
        layer.block{l-1}.delta = layer.block{l}.delta;
        if layer.down_sampling.option == true
            smp = layer.block{layer.down_sampling.layer};
            smp.delta = layer.block{l}.delta;
            pre_delta_res = SMP_backpropagation(smp, size(pre_X{1}(:,:,1)));
        else
            pre_delta_res = layer.block{l}.delta;
        end
    end
end

[layer.block{1}, pre_delta] = CN_backpropagation(layer.block{1}, pre_X);

for i = 1 : length(pre_delta)
    
    pre_delta{i} = pre_delta{i} + pre_delta_res{i};

end

end
