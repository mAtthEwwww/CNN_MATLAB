function CNN = CNN_backpropagation( CNN , target )
% CNN_backpropagation.m
% backpropagation with target
%
% Inputs:
%       target  is a bool matrix, each row is a one-hot label of a sample 
%       CNN  is a array of cell, each cell is a layer of CNN
%
% Outputs:
%       CNN  is a array of cell


% extract the number of sample
[N, ~] = size(target);

% extract the number of layer
L = length(CNN);

% calculate the delta of the final layer
CNN{L}.delta = (CNN{L}.X - target) / N;

% calculate the delta from back to front, layer by layer
for l = L : -1 : 2
    % judge the type of layer l
    if strcmp(CNN{l}.type, 'convolution')
        % call the back propagation function accordingly
        [CNN{l}, CNN{l-1}.delta] = BP_convolution_layer(CNN{l}, CNN{l-1}.X);

    elseif strcmp(CNN{l}.type, 'residual_block')
        [CNN{l}, CNN{l-1}.delta] = BP_residual_block(CNN{l}, CNN{l-1}.X);

    elseif strcmp(CNN{l}.type, 'batch_normalization')
        [CNN{l}, CNN{l-1}.delta] = BP_batch_normalization_layer(CNN{l});

    elseif strcmp( CNN{l}.type, 'sampling' )
        CNN{l-1}.delta = BP_sampling_layer(CNN{l}, CNN{l-1}.map_size);

    elseif strcmp(CNN{l}.type, 'full_connection')
        [CNN{l}, CNN{l-1}.delta] = BP_full_connection_layer(CNN{l}, CNN{l-1}.X, CNN{l-1}.isTensor);

    elseif strcmp( CNN{l}.type, 'activation' )
        CNN{l-1}.delta = BP_activation_layer(CNN{l}, CNN{l-1}.X);

    else
        error('layer type wrong');
    end
end
        
end
