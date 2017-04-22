% see more detail in CIFAR_main.m

clear
clc

rand( 'state' , 0 );
randn( 'state' , 0 );

addpath 'datasets/MNIST_dataset'
addpath 'CNN_frame'
addpath 'CNN_frame/util'
addpath 'CNN_frame/layer_constructor'
addpath 'CNN_frame/layer_calculator'

[ train , test ] = load_MNIST_2d(3);
train_input = { train.X };
train_target = train.T;
test_input = { test.X };
test_target = test.T;
validation_size = 1000;
[validation_input, validation_target] = get_mini_batch(test_input, test_target, 1 : validation_size);
clear train
clear test

tr_config.learning_rate = 0.01;
tr_config.half_life = 5;
tr_config.momentum = 0;
tr_config.weight_decay = 0.0001;
tr_config.batch_size = 50;
tr_config.validate_interval = 200;
tr_config.max_epochs = 15;
tr_config.cost_function = 'cross_entropy';
tr_config.threshold = 0.002;

input_channel = length(train_input);
input_size = size(train_input{1}(:, :, 1));


CNN{ 1 }.output = input_channel;
CNN{ 1 }.map_size = input_size;

CNN{ 2 }.type = 'convolution';
CNN{ 2 }.weight.filler.type = 'xavier';
CNN{ 2 }.weight.learning_rate = 1;
CNN{ 2 }.weight.shape = [5, 5];
CNN{ 2 }.bias.option = false;
CNN{ 2 }.bias.learning_rate = 2;
CNN{ 2 }.output = 20;
CNN{ 2 }.zero_padding.option = false;

CNN{ 3 }.type = 'batch_normalization';
CNN{ 3 }.BN_decay = 0.95;

CNN{ 4 }.type = 'sampling';
CNN{ 4 }.sampling.type = 'max';
CNN{ 4 }.sampling.shape = [2, 2];
CNN{ 4 }.sampling.stride = [2, 2];

CNN{ 5 }.type = 'convolution';
CNN{ 5 }.weight.filler.type = 'xavier';
CNN{ 5 }.weight.learning_rate = 1;
CNN{ 5 }.weight.shape = [5, 5];
CNN{ 5 }.bias.option = false;
CNN{ 5 }.bias.learning_rate = 2;
CNN{ 5 }.output = 40;
CNN{ 5 }.zero_padding.option = false;

CNN{ 6 }.type = 'batch_normalization';
CNN{ 6 }.BN_decay = 0.95;

CNN{ 7 }.type = 'sampling';
CNN{ 7 }.sampling.type = 'max';
CNN{ 7 }.sampling.shape = [2, 2];
CNN{ 7 }.sampling.stride = [2, 2];

CNN{ 8 }.type = 'full_connection';
CNN{ 8 }.weight.filler.type = 'xavier';
CNN{ 8 }.weight.learning_rate = 1;
CNN{ 8 }.bias.learning_rate = 2;
CNN{ 8 }.dropout.option = false;
CNN{ 8 }.output = 200;

CNN{ 9 }.type = 'activation';
CNN{ 9 }.activation = 'relu';

CNN{ 10 }.type = 'full_connection';
CNN{ 10 }.weight.filler.type = 'xavier';
CNN{ 10 }.weight.learning_rate = 1;
CNN{ 10 }.bias.learning_rate = 2;
CNN{ 10 }.dropout.option = false;
CNN{ 10 }.output = 10;

CNN{ 11 }.type = 'activation';
CNN{ 11 }.activation = 'softmax';

CNN = CNN_initialization( CNN );

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check( train_input , train_target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );

tic;
CNN = CNN_train( train_input, train_target, validation_input, validation_target, tr_config, CNN );
train_time = toc;

[accuracy, confusion_matrix] = CNN_test( test_input, test_target, CNN );

disp(accuracy)
