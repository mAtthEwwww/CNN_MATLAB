function CNN = CNN_feedforward( CNN , X , isTrain )
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

% store the input set in the input layer

CNN{1}.X = X;

% do the feedforward layer by layer
for l = 2 : length(CNN)
    % if layer l is convolution layer, then do the convolution
    if strcmp(CNN{l}.type, 'convolution')
        CNN{l} = CN_feedforward(CNN{l}, CNN{l-1}.X);

    elseif strcmp(CNN{l}.type, 'residual_block')
        CNN{l} = RESBLK_feedforward(CNN{l}, CNN{l-1}.X, isTrain);
        
    % if layer l is batch normalization layer
    elseif strcmp(CNN{l}.type, 'batch_normalization')
        CNN{l} = BN_feedforward(CNN{l}, CNN{l-1}.X, isTrain);

    % if layer l is sampling layer, then do the down sampling
    elseif strcmp(CNN{l}.type, 'sampling')
        CNN{l} = SMP_feedforward(CNN{l}, CNN{l-1}.X);

    % if layer l is full connection layer, then do the inner product
    elseif strcmp(CNN{l}.type, 'full_connection')
        CNN{l} = FC_feedforward(CNN{l}, CNN{l-1}.X, CNN{l-1}.isTensor, isTrain);

    % if layer l is activation layer, then do the activation
    elseif strcmp(CNN{l}.type, 'activation')
        CNN{l} = ACT_feedforward(CNN{l}, CNN{l-1}.X);

    else
        error('layer type wrong')
    end
end

end
