function [ CNN , result ]  = CNN_train( train_input , train_target , validation_input , validation_target , config , CNN )
% CNN_train.m
% CNN training function, optimizing with stochastic gradient descent

% Inputs:
%       train_input  is a array of cell, each cell is a 3-order tensor,
%                   which represents a color channel
%       train_target  is a bool matrix, each row represents a one-hot 
%                   label of a sample
%       validation_input  is a array of cell
%       validation_target  is a bool matrix
%       config  is a struct contains training strategy:
%               learning_rate, learning rate of gradient descent
%               half_life, half life(epoch) of the learning rate
%               weight_decay, weight decay parameter,or L2 norm regularization
%               max_epochs, the maximum number of epochs
%               momentum, the momentum of gradient
%               batch_size, the size of mini batch
%               validate_interval, the iteration interval between successive validation
%               cost_function, the cost function, or loss function, or object funtion
%               threshold, the threshold of cost function value
%       CNN  is an array of cell, each cell is a layer of CNN
%
% Outputs:
%       CNN  is an array of cell


% extract the training strategy
learning_rate = config.learning_rate;
half_life = config.half_life;
weight_decay = config.weight_decay;
max_epochs = config.max_epochs;
momentum = config.momentum;
batch_size = config.batch_size;
validate_interval = config.validate_interval;
threshold = config.threshold;

% extract the sample size
[ ~ , ~ , N ] = size(train_input{1});

% calculate the iteration number of a epoch
iterations = floor( N / batch_size );

% construct an empty array for cost value of every iteration
cost = zeros( 1 , max_epochs * iterations );

% construct an empty array for moving average of the cost value
average_cost = zeros( 1 , max_epochs * iterations );

% construct an empty array for validation accuracy
validate_accuracy = zeros( 2 , floor( max_epochs * iterations / validate_interval )+1 );

% initializing the iteration counter
iter = 0;

% initializing the epoch counter
epochs = 0;

% initializing the validation counter
validated = 1;

% initializing the training indicator
go_on = 1;

% calculate the parameter of exponential decay
k = -log(2) / (iterations * half_life );

% outer loop of training
while epochs < max_epochs && go_on
    % epoch counter increasing
    epochs = epochs + 1;
    
    % shuffle the training set
    shuffle = randperm( N );

    % initializing the counter inner loop
    sub_iter = 0;

    % inner loop of training
    while sub_iter < iterations && go_on
        
        % iteration counter increasing
        iter = iter + 1;
        
        % inner loop counter increasing
        sub_iter = sub_iter + 1;

        % learning rate exponential deacy
        lr = learning_rate * exp(k * iter);
        
        % extract a mini-batch from the training set
        if sub_iter < iterations
            [batch_input, batch_target] = get_mini_batch(train_input, train_target, shuffle((sub_iter-1)*batch_size+1 : sub_iter*batch_size));
        else
            [batch_input, batch_target] = get_mini_batch(train_input, train_target, shuffle((sub_iter-1)*batch_size+1 : end));
        end

        % feedforward with input set of the mini-batch 
        CNN = CNN_feedforward(CNN, batch_input, true, batch_target);

        % backpropagation with the target set of the mini-batch
        % and calculate the gradient
        CNN = CNN_backpropagation(CNN, batch_target);

        % update the weight and bias
        % with graient, momentum, learning rate and weight decay rate
        CNN = CNN_gradient_descent(CNN, lr, momentum, weight_decay);

        % if condition is satisfied, do the validate
        if mod(iter, validate_interval) == 0

            % validated counter increasing
            validated = validated + 1;

            % test CNN with validation set
            [accuracy, ~] = CNN_test(validation_input, validation_target, CNN);

            % store the validation result
            validate_accuracy(1, validated) = iter;

            validate_accuracy(2, validated) = accuracy;
        end

        % extract the cost value
        cost(iter) = CNN{length(CNN)}.cost_value;

        % calculate the moving average of cost value
        average_cost(iter) = mean(cost(max(1, iter - iterations) : iter));

        % display the running detail
        fprintf('Accuracy: %.4f, LR: %.7f, EPO: %i/%i, ITR: %i/%i, AVG: %f, CUR: %f\n' , validate_accuracy(2, validated), lr, epochs, max_epochs, sub_iter, iterations, average_cost(iter), cost(iter));

        % check the iteration stop condition
        if average_cost(iter) < threshold
            % if stop condition is satisfied, then stop
            go_on = 0;
        end
    end
end

% new figure
figure

% draw the curve of cost moving average
% draw the curve of validation accuracy
[ img, ~, ~ ] = plotyy( 1 : iter, average_cost( 1 : iter ) , validate_accuracy( 1 , 1 : validated ) , validate_accuracy( 2 , 1 : validated ) , @plot );

% draw the y label
set( get( img(1), 'ylabel' ), 'string', 'cost value', 'fontsize', 16 );
set( get( img(2), 'ylabel' ), 'string', 'AC', 'fontsize', 16 );

% draw the x label
xlabel( 'iterations', 'fontsize', 16 );

% store the last moving average value in the first layer
result.cost = average_cost(iter);

% store the epochs number in the first layer
result.cost = epochs;

end
