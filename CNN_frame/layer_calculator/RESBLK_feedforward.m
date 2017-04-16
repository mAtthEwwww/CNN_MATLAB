function layer = RESBLK_feedforward( layer , X , isTrain )

if nargin < 3
    isTrain = false;
end

[~, ~, N] = size(X{1});

layer.block{1} = CN_feedforward(layer.block{1}, X);

for l = 2 : length(layer.block)
    if strcmp(layer.block{l}.type, 'convolution')
        layer.block{l} = CN_feedforward(layer.block{l}, layer.block{l-1}.X);

    % if layer l is batch normalization layer
    elseif strcmp(layer.block{l}.type, 'batch_normalization')
        layer.block{l} = BN_feedforward(layer.block{l}, layer.block{l-1}.X, isTrain);

    % if layer l is sampling layer, then do the down sampling
    elseif strcmp(layer.block{l}.type, 'sampling')
        layer.block{l} = SMP_feedforward(layer.block{l}, layer.block{l-1}.X);

    elseif strcmp(layer.block{l}.type, 'activation')
        layer.block{l} = ACT_feedforward(layer.block{l}, layer.block{l-1}.X);

    elseif strcmp(layer.block{l}.type, 'residual')
        if layer.down_sampling.option == true
            smp = SMP_feedforward(layer.block{layer.down_sampling.layer}, X);
            layer.block{l} = RES_feedforward(layer.block{l}, layer.block{l-1}.X, smp.X);
        else
            layer.block{l} = RES_feedforward(layer.block{l}, layer.block{l-1}.X, smp.X);
        end
    end

end

layer.X = layer.block{length(layer.block)}.X;

end
