% see more detail in CIFAR_main.m

clear
clc

rand( 'state' , 0 );
randn( 'state' , 0 );

addpath 'CNN_frame'
addpath 'CNN_frame/util'
addpath 'CNN_frame/layer_constructor'
addpath 'CNN_frame/layer_calculator'
addpath 'datasets/MNIST_dataset'


%% -------------------prepare data set----------------------

[ train , test ] = load_MNIST_2d(3);
train_input = {train.X};
train_target = train.T;
test_input = {test.X};
test_target = test.T;
validation_size = 1000;
[validation_input, validation_target] = get_mini_batch(test_input, test_target, 1 : validation_size);
clear train
clear test

input_size = size(train_input{1}(:, :, 1));
input_channel = length(train_input);
test_batch_size = 1000;


%% -------------configure the training strategy-------------

tr_config.learning_rate = 0.02;
tr_config.half_life = 10;
tr_config.momentum = 0;
tr_config.weight_decay = 0.0001;
tr_config.batch_size = 30;
tr_config.validate_interval = 200;
tr_config.max_epochs = 5;
tr_config.cost_function = 'cross_entropy';
tr_config.threshold = 0.002;


%% -------------define the structure of CNN----------------

l = 1;
CNN{l}.map_size = input_size;
CNN{l}.output = input_channel;

l = l + 1;
CNN{l}.type = 'residual_block';
CNN{l}.weight.filler.type = 'xavier';
CNN{l}.weight.learning_rate = 1;
CNN{l}.BN_decay = 0.99;
CNN{l}.sampling.option = true;
CNN{l}.sampling.type = 'max';
CNN{l}.sampling.shape = [2, 2];
CNN{l}.sampling.stride = [2, 2];
CNN{l}.output = 10;

l = l + 1;
CNN{l}.type = 'residual_block';
CNN{l}.weight.filler.type = 'xavier';
CNN{l}.weight.learning_rate = 1;
CNN{l}.BN_decay = 0.99;
CNN{l}.sampling.option = true;
CNN{l}.sampling.type = 'max';
CNN{l}.sampling.shape = [2, 2];
CNN{l}.sampling.stride = [2, 2];
CNN{l}.output = 20;

l = l + 1;
CNN{l}.type = 'residual_block';
CNN{l}.weight.filler.type = 'xavier';
CNN{l}.weight.learning_rate = 1;
CNN{l}.BN_decay = 0.99;
CNN{l}.sampling.option = true;
CNN{l}.sampling.type = 'max';
CNN{l}.sampling.shape = [2, 2];
CNN{l}.sampling.stride = [2, 2];
CNN{l}.output = 40;

l = l + 1;
CNN{l}.type = 'full_connection';
CNN{l}.weight.filler.type = 'xavier';
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.learning_rate = 2;
CNN{l}.dropout.option = true;
CNN{l}.dropout.rate = 0.5;
CNN{l}.output = 80;

l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';

l = l + 1;
CNN{l}.type = 'full_connection';
CNN{l}.weight.filler.type = 'xavier';
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.learning_rate = 2;
CNN{l}.dropout.option= false;
CNN{l}.output = 10;

l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'softmax';


%% -------------initializaing the CNN-------------------

CNN = CNN_initialization(CNN);


%% --------------check the gradient---------------------

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check(train_input, train_target, CNN, tr_config.cost_function, check_size, epsilon, tolerance);


%% -----------------train the CNN-----------------------

tic;

CNN = CNN_train(train_input, train_target, validation_input, validation_target, tr_config, CNN);

result.train_time = toc;


%% ----------------test the CNN------------------------

[result.accuracy, result.confusion_matrix] = CNN_test(test_input, test_target, CNN, test_batch_size);


%% ------------save the running log and figure---------

writing_log(CNN, result, tr_config);


