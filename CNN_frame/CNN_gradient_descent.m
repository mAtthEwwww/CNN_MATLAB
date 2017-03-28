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
                CNN{l}.m_grad_kernel{i}{j} = momentum * CNN{l}.m_grad_kernel{i}{j} + learning_rate * CNN{l}.weight_learning_rate * ( CNN{l}.grad_kernel{i}{j} + weight_decay * CNN{l}.kernel{i}{j} );
		CNN{l}.kernel{i}{j} = CNN{l}.kernel{i}{j} - CNN{l}.m_grad_kernel{i}{j};
            end
	    CNN{l}.m_grad_bias{j} = momentum * CNN{l}.m_grad_bias{j} + learning_rate * CNN{l}.bias_learning_rate * CNN{l}.grad_bias{j};
            CNN{l}.bias{j} = CNN{l}.bias{j} - CNN{l}.bias{j} - CNN{l}.m_grad_bias{j};
        end
    elseif strcmp( CNN{l}.type, 'full_connection' )
        CNN{l}.m_grad_W = momentum * CNN{l}.m_grad_W + learning_rate * CNN{l}.weight_learning_rate * ( CNN{l}.grad_W + weight_decay * CNN{l}.W );
        CNN{l}.W = CNN{l}.W - CNN{l}.m_grad_W;
	CNN{l}.m_grad_bias = momentum * CNN{l}.m_grad_bias + learning_rate * CNN{l}.bias_learning_rate * CNN{l}.grad_bias;
        CNN{l}.bias = CNN{l}.bias - CNN{l}.m_grad_bias;
    end
end

end
