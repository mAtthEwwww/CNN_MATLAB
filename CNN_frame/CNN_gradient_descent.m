function CNN =  CNN_gradient_descent ( CNN , momentum , learning_rate , weight_decay )
% CNN_gradient_descent.m
% function for mini-batch gradient descent, with momentum
%
% Inputs:
%       CNN  is an array of cell, each cell is a layer of CNN
%       momentum  is a number between 0 and 1, presents the rate of gradient momentum
%
% Outputs:
%       CNN  is an array of cell

% update the weight and bias layer by layer
for l = 2 : length( CNN )
    if strcmp( CNN{l}.type, 'convolution' )
        for j = 1 : CNN{l}.output
            for i = 1 : CNN{l-1}.output
                CNN{l}.weight.momentum{j,i} = momentum * CNN{l}.weight.momentum{j,i} + learning_rate * CNN{l}.weight.learning_rate * (CNN{l}.weight.grad{j,i} + weight_decay * CNN{l}.weight.kernel{j,i});
                CNN{l}.weight.kernel{j,i} = CNN{l}.weight.kernel{j,i} - CNN{l}.weight.momentum{j,i};
            end
            if CNN{l}.bias.option == true
	            CNN{l}.bias.momentum{j} = momentum * CNN{l}.bias.momentum{j} + learning_rate * CNN{l}.bias.learning_rate * CNN{l}.bias.grad{j};
                CNN{l}.bias.b{j} = CNN{l}.bias.b{j} - CNN{l}.bias.momentum{j};
            end
        end
    elseif strcmp(CNN{l}.type, 'batch_normalization')
        for j = 1 : CNN{l}.output
            CNN{l}.gamma.momentum{j} = momentum * CNN{l}.gamma.momentum{j} + learning_rate * CNN{l}.gamma.grad{j};% learning rate of gamma or beta is considerable
            CNN{l}.gamma.g{j} = CNN{l}.gamma.g{j} - CNN{l}.gamma.momentum{j};
            CNN{l}.beta.momentum{j} = momentum * CNN{l}.beta.momentum{j} + learning_rate * CNN{l}.beta.grad{j};
            CNN{l}.beta.b{j} = CNN{l}.beta.b{j} - CNN{l}.beta.momentum{j};
        end
    elseif strcmp( CNN{l}.type, 'full_connection' )
        CNN{l}.weight.momentum = momentum * CNN{l}.weight.momentum + learning_rate * CNN{l}.weight.learning_rate * (CNN{l}.weight.grad + weight_decay * CNN{l}.weight.W );
        CNN{l}.weight.W = CNN{l}.weight.W - CNN{l}.weight.momentum;
	    CNN{l}.bias.momentum = momentum * CNN{l}.bias.momentum + learning_rate * CNN{l}.bias.learning_rate * CNN{l}.bias.grad;
        CNN{l}.bias.b = CNN{l}.bias.b - CNN{l}.bias.momentum;
    end
end

end
