function block = FF_residual_block( block , X , isTrain )

if nargin < 3
    isTrain = false;
end

[~, ~, N] = size(X{1});

block.layer{1} = FF_convolution_layer(block.layer{1}, X);

for l = 2 : length(block.layer)
    if strcmp(block.layer{l}.type, 'convolution')
        block.layer{l} = FF_convolution_layer(block.layer{l}, block.layer{l-1}.X);

    % if block l is batch normalization block
    elseif strcmp(block.layer{l}.type, 'batch_normalization')
        block.layer{l} = FF_batch_normalization_layer(block.layer{l}, block.layer{l-1}.X, isTrain);

    % if block l is sampling block, then do the down sampling
    elseif strcmp(block.layer{l}.type, 'sampling')
        block.layer{l} = FF_sampling_layer(block.layer{l}, block.layer{l-1}.X);

    elseif strcmp(block.layer{l}.type, 'activation')
        block.layer{l} = FF_activation_layer(block.layer{l}, block.layer{l-1}.X);

    elseif strcmp(block.layer{l}.type, 'short_cut')
        block.layer{l}.X = block.layer{l-1}.X;
        if block.sampling.option == true
            shortcut_samp = block.layer{block.sampling.block};
            shortcut_samp = FF_sampling_layer(shortcut_samp, X);
            for i = 1 : length(X)
                block.layer{l}.X{i} = block.layer{l}.X{i} + shortcut_samp.X{i};
            end
        else
            for i = 1 : length(X)
                block.layer{l}.X{i} = block.layer{l}.X{i} + X{i};
            end
        end
    end

end

block.X = block.layer{length(block.layer)}.X;

end
