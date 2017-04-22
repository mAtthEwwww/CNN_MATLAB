function CNN =  CNN_gradient_descent( CNN , lr , momentum ,weight_decay)
% CNN_gradient_descent.m
% function for mini-batch gradient descent, with momentum
%
% Inputs:
%       CNN  is an array of cell, each cell is a layer of CNN
%       config is a struct, contains the training strategy
%
% Outputs:
%       CNN  is an array of cell


% update the parameter layer by layer
for l = 2 : length(CNN)
    % judge the type of layer l
    if strcmp(CNN{l}.type, 'convolution')
        % call the gradient descent function accrodingly
        CNN{l} = GD_convolution_layer(CNN{l}, lr, momentum, weight_decay);

    elseif strcmp(CNN{l}.type, 'residual_block')
        CNN{l} = GD_residual_block(CNN{l}, lr, momentum, weight_decay);

    elseif strcmp(CNN{l}.type, 'batch_normalization')
        CNN{l} = GD_batch_normalization_layer(CNN{l}, lr, momentum);

    elseif strcmp(CNN{l}.type, 'full_connection')
        CNN{l} = GD_full_connection_layer(CNN{l}, lr, momentum, weight_decay);

    end
end

end
