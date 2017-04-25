% CIFAR_LeNet.m

clear
clc

rand( 'state' , 0 );                                        % configure random seed
randn( 'state' , 0 );

addpath 'CNN_frame'                                         %add the path
addpath 'CNN_frame/util'
addpath 'CNN_frame/layer_constructor'
addpath 'CNN_frame/layer_calculator'
addpath 'datasets/CIFAR_10_dataset'


%% -------------------prepare data set----------------------

% Load the cifar-10 data set, with with pre-treated type 2.
% Train and test are struct, both of which contain a input set X and a output set T.
% The input set X is an array of cell (X{1}, X{2}, X{3}), each cell contains a 3-order tensor. Each tensor (height, width, sample-size) presents a color channel (RGB)
% The output set T is a label matrix, each row is a one-hot label vector
[train, test] = load_cifar(2);

validation_size = 2000;                                     % set the size of validate set

train_input = train.X;                                      % train set
train_target = train.T;
clear train;

test_input = test.X;                                        % test set
test_target = test.T;
clear test;

[validation_input, validation_target] = get_mini_batch(test_input, test_target, 1 : validation_size);
                                                            % divide a validate set from test set
input_size = size(train_input{1}(:, :, 1));                 % the shape of each input image (height, width)
input_channel = length(train_input);                        % the number of color channel
test_batch_size = 2000;                                     % the size of a testing batch


%% -------------configure the training strategy-------------

tr_config.learning_rate = 0.05;                             % initial global learning rate
tr_config.half_life = 5;                                    % learning rate exponential decay with specific half-life
tr_config.momentum = 0.9;                                   % momentum (or exponential moving average) of gradient
tr_config.weight_decay = 0.00001;                           % the ratio of weight decay
tr_config.batch_size = 30;                                  % the batch-size of stochastic gradient descent
tr_config.validate_interval = 5000;                         % the interval (iteration) between successive validation
tr_config.max_epochs = 10;                                  % the maximum number of epochs
tr_config.threshold = 0.05;                                 % the threshold of cost value


%% -------------define the structure of CNN----------------

% input layer
l = 1;
CNN{1}.type = 'input';
CNN{l}.output = input_channel;
CNN{l}.map_size = input_size;

% convolution layer
l = l + 1;
CNN{l}.type = 'convolution';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.001;
CNN{l}.weight.shape = [3, 3];                               % when turn on zero padding, make sure the kernel size is odd
CNN{l}.zero_padding.option = true;
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.output = 10;


%% batch normalization layer
%l = l + 1;
%CNN{l}.type = 'batch_normalization';
%%the moving average decay ratio of batch normalization
%CNN{l}.BN_decay = 0.99;


% sampling layer
l = l + 1;
CNN{l}.type = 'sampling';
CNN{l}.sampling.type = 'max';
CNN{l}.sampling.shape = [3, 3];
CNN{l}.stride = [2, 2];


% activation layer
l = l + 1
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';


% convolution layer
l = l + 1;
CNN{l}.type = 'convolution';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.01;
CNN{l}.weight.learning_rate = 1;
CNN{l}.weight.shape = [3 ,3];
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.zero_padding.option = true;
CNN{l}.output = 20;


%% batch normalization layer
%l = l + 1;
%CNN{l}.type = 'batch_normalization';
%CNN{l}.BN_decay = 0.99;


% activation layer
l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';


% sampling layer
l = l + 1;
CNN{l}.type = 'sampling';
CNN{l}.sampling.type = 'average';
CNN{l}.sampling.shape = [3, 3];
CNN{l}.sampling.stride = [2, 2];


% convolution layer
l = l + 1;
CNN{l}.type = 'convolution';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.01;
CNN{l}.weight.learning_rate = 1;
CNN{l}.weight.shape = [3, 3];
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.zero_padding.option = true;
CNN{l}.output = 40;


%% batch normalization layer
%l = l + 1;
%CNN{l}.type = 'batch_normalization';
%CNN{l}.BN_decay = 0.99;



% activation layer
l = l + 1;
CNN{ l }.type = 'activation';
CNN{ l }.activation = 'relu';


% sampling layer
l = l + 1;
CNN{l}.type = 'sampling';
CNN{l}.sampling.type = 'average';
CNN{l}.sampling.shape = [3, 3];
CNN{l}.sampling.stride = [2, 2];


% full connection layer
l = l + 1;
CNN{l}.type = 'full_connection';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.1;
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.learning_rate = 2;
CNN{l}.dropout.option = true;
CNN{l}.dropout.rate = 0.5;
CNN{l}.output = 32;


% activation layer
l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';


% full connection layer
l = l + 1;
CNN{l}.type = 'full_connection';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.1;
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.learning_rate = 2;
CNN{l}.dropout.option = false;
CNN{l}.output = 10;


% output layer
l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'softmax';
CNN{l}.cost_function = 'cross_entropy';



%% -------------initializaing the CNN-------------------

CNN = CNN_initialization(CNN);


%% --------------check the gradient---------------------

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check(train_input, train_target, CNN, tr_config.cost_function, check_size, epsilon, tolerance);


%% -----------------train the CNN-----------------------

tic;

[CNN, result] = CNN_train(train_input, train_target, validation_input, validation_target, tr_config, CNN);

result.train_time = toc;

%% ----------------test the CNN------------------------

[result.accuracy, result.confusion_matrix] = CNN_test(test_input, test_target, CNN, test_batch_size);



%% ------------save the running log and figure---------

writing_log(CNN, result, tr_config);
