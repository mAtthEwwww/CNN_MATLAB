function CNN = CNN_train( train_input , train_target , validation_input , validation_target , config , CNN )
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


learning_rate = config.learning_rate;
half_life = config.half_life;
weight_decay = config.weight_decay;
max_epochs = config.max_epochs;
momentum = config.momentum;
batch_size = config.batch_size;
validate_interval = config.validate_interval;
cost_function = str2func( config.cost_function );
threshold = config.threshold;
[ ~ , ~ , N ] = size( train_input{ 1 } );

iterations = floor( N / batch_size );               %计算每个epoch中需要迭代的次数
cost = zeros( 1 , max_epochs * iterations );        %申请一个数组，存放每次迭代的损失函数值
average_cost = zeros( 1 , max_epochs * iterations );%申请一个数值，存放最近一个epoch内的平均损失函数值
validate_accuracy = zeros( 2 , floor( max_epochs * iterations / validate_interval )+1 ); %申请一个数组，存放每次验证集的测试正确率
iter = 0;                                           %设置计数器，记录总的迭代次数
epochs = 0;                                         %设置计数器，记录整个训练集合的训练次数
validated = 1;                                      %设置计数器，记录对验证集的进行测试的次数
go_on = 1;                                          %设置一个迭代停止的标记，如果标记为0，则停止迭代
k = -log(2) / (iterations * half_life );            %根据半衰期计算指数衰减的参数
while epochs < max_epochs && go_on
    epochs = epochs + 1;
    
    shuffle = randperm( N );                       %每次对整个训练集进行训练时，先打乱顺序，产生一个随机序列索引
    sub_iter = 0;                                   %设置计数器，记录在这个epoch内的迭代次数
    while sub_iter < iterations && go_on
        
        iter = iter + 1;
        sub_iter = sub_iter + 1;
        lr = learning_rate * exp(k * iter);
        
        if sub_iter < iterations
            [batch_input, batch_target] = get_mini_batch(train_input, train_target, shuffle((sub_iter-1)*batch_size+1 : sub_iter*batch_size));
        else
            [batch_input, batch_target] = get_mini_batch(train_input, train_target, shuffle((sub_iter-1)*batch_size+1 : end));
        end
        
        CNN = CNN_feedforward(batch_input, CNN);
        CNN = CNN_backprogagation(batch_target, CNN);
        CNN = CNN_gradient_descent(CNN, momentum, lr, weight_decay);
        
        if mod(iter, validate_interval) == 0
            validated = validated + 1;
            [accuracy, ~] = CNN_test(validation_input, validation_target, CNN);
            validate_accuracy(1, validated) = iter;
            validate_accuracy(2, validated) = accuracy;
        end
        cost(iter) = cost_function(CNN{length(CNN)}.X, batch_target);        
        average_cost(iter) = mean(cost(max(1, iter - iterations) : iter));
        fprintf('Accuracy: %.4f, LR: %.7f, EPO: %i/%i, ITR: %i/%i, AVG: %f, CUR: %f\n' , validate_accuracy(2, validated), lr, epochs, max_epochs, sub_iter, iterations, average_cost(iter), cost(iter));
        if average_cost(iter) < threshold
            go_on = 0;
        end
    end
end

figure

[ img, ~, ~ ] = plotyy( 1 : iter, average_cost( 1 : iter ) , validate_accuracy( 1 , 1 : validated ) , validate_accuracy( 2 , 1 : validated ) , @plot );
set( get( img(1), 'ylabel' ), 'string', 'cost value', 'fontsize', 16 );
set( get( img(2), 'ylabel' ), 'string', 'AC', 'fontsize', 16 );
xlabel( 'iterations', 'fontsize', 16 );

CNN{1}.cost = average_cost( iter );
CNN{1}.epochs = epochs;

end
