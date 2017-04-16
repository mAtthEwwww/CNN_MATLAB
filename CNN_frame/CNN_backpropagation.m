function CNN = CNN_backpropagation( target , CNN )
% CNN_backpropagation.m
% backpropagation with target
%
% Inputs:
%       target  is a bool matrix, each row is a one-hot label of a sample 
%       CNN  is a array of cell, each cell is a layer of CNN
%
% Outputs:
%       CNN  is a array of cell


% extract the sample size
[ ~, ~, N ] = size(CNN{1}.X{1});

% extract the number of layer
L = length( CNN );

% calculate the delta of the final layer
CNN{L}.delta = (CNN{L}.X - target) / N;

% calculate the delta from back to front, layer by layer
for l = L : -1 : 2
    % if layer l is convolution layer, do the transpose convolution (or deconvolution)
    if strcmp(CNN{l}.type, 'convolution')
        [CNN{l}, CNN{l-1}.delta] = CN_backpropagation(CNN{l}, CNN{l-1}.X);

    elseif strcmp(CNN{l}.type, 'residual_block')
        [CNN{l}, CNN{l-1}.delta] = RESBLK_backpropagation(CNN{l}, CNN{l-1}.X);%%%%%%%%

    elseif strcmp(CNN{l}.type, 'batch_normalization')
        [CNN{l}, CNN{l-1}.delta] = BN_backpropagation(CNN{l});

    % if layer l is sampling layer, do the up sampling
    elseif strcmp( CNN{l}.type, 'sampling' )
        CNN{l-1}.delta = SMP_backpropagation(CNN{l}, CNN{l-1}.map_size);

    % if layer l is full connection layer
    elseif strcmp(CNN{l}.type, 'full_connection')
        [CNN{l}, CNN{l-1}.delta] = FC_backpropagation(CNN{l}, CNN{l-1}.X, CNN{l-1}.isTensor);

    % if layer l is activation layer, then multiply delta by derived activation
    elseif strcmp( CNN{l}.type, 'activation' )
        % transfer the string of activation to function handle of derivatives
        CNN{l-1}.delta = ACT_backpropagation(CNN{l}, CNN{l-1}.X);

    else
        error('layer type wrong');
    end
end
        
end
