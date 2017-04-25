function CNN = CNN_feedforward( CNN , X , isTrain , T )
% CNN_feedforward.m
% the feedforward function of CNN
% Inputs:
%       X  is an array of cell, each cell contains a 3-order tensor
%               (height x width x sample-size) and represent a color channel 
%       CNN  is an array of cell, each cell is a layer of CNN
% Outputs:
%       CNN  is an array of cell, the feedforward result is stored in the last layer (CNN{length(CNN)}.X)

if nargin < 3
    isTrain = false;
end

% extract the sample size (the number of sample)
[ ~ , ~ , N ] = size(X{1});

% store the input set in the first layer
CNN{1}.X = X;

% execute feedforward layer by layer
for l = 2 : length(CNN)
    % judge the type of layer l
    if strcmp(CNN{l}.type, 'convolution')
        % call the feedforward function accordingly
        CNN{l} = FF_convolution_layer(CNN{l}, CNN{l-1}.X);

    elseif strcmp(CNN{l}.type, 'residual_block')
        CNN{l} = FF_residual_block(CNN{l}, CNN{l-1}.X, isTrain);
        
    elseif strcmp(CNN{l}.type, 'batch_normalization')
        CNN{l} = FF_batch_normalization_layer(CNN{l}, CNN{l-1}.X, isTrain);

    elseif strcmp(CNN{l}.type, 'sampling')
        CNN{l} = FF_sampling_layer(CNN{l}, CNN{l-1}.X);

    elseif strcmp(CNN{l}.type, 'full_connection')
        CNN{l} = FF_full_connection_layer(CNN{l}, CNN{l-1}.X, CNN{l-1}.isTensor, isTrain);

    elseif strcmp(CNN{l}.type, 'activation')
        CNN{l} = FF_activation_layer(CNN{l}, CNN{l-1}.X);

    else
        error('layer type wrong')
    end
end

if isTrain

    cost_function = str2func(CNN{l}.cost_function);

    CNN{l}.cost_value = cost_function(CNN{l}.X, T);

end

end
