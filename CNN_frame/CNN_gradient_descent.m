function CNN =  CNN_gradient_descent( CNN , learning_rate , momentum ,weight_decay)
% CNN_gradient_descent.m
% function for mini-batch gradient descent, with momentum
%
% Inputs:
%       CNN is an array of cell, each cell is a layer of the network
%       learning_rate is a postive number, which is the global learning rate
%       momentum is a number between 0 and 1, which is the momentum rate of GD
%       weight is a postive number, which is the ratio of L2 normalization
%
% Outputs:
%       CNN is an array of cell, each cell is a layer of the network


% update the parameter layer by layer
for l = 2 : length(CNN)
    % judge the type of layer l
    if strcmp(CNN{l}.type, 'convolution')
        % call the gradient descent function accrodingly
        CNN{l} = GD_convolution_layer(CNN{l}, learning_rate, momentum, weight_decay);

    elseif strcmp(CNN{l}.type, 'residual_block')
        CNN{l} = GD_residual_block(CNN{l}, learning_rate, momentum, weight_decay);

    elseif strcmp(CNN{l}.type, 'batch_normalization')
        CNN{l} = GD_batch_normalization_layer(CNN{l}, learning_rate, momentum);

    elseif strcmp(CNN{l}.type, 'full_connection')
        CNN{l} = GD_full_connection_layer(CNN{l}, learning_rate, momentum, weight_decay);

    end
end

end
