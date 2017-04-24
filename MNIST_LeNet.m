% see more detail in CIFAR_LeNet.m

clear
clc

rand( 'state' , 0 );
randn( 'state' , 0 );

addpath 'CNN_frame'
addpath 'CNN_frame/util'
addpath 'CNN_frame/layer_constructor'
addpath 'CNN_frame/layer_calculator'

addpath 'datasets/MNIST_dataset'
[ train , test ] = load_MNIST_2d(3);
train_input = {train.X};
train_target = train.T;
test_input = {test.X};
test_target = test.T;
validation_size = 1000;
[validation_input, validation_target] = get_mini_batch(test_input, test_target, 1 : validation_size);
clear train
clear test

tr_config.learning_rate = 0.4;
tr_config.half_life = 2;
tr_config.momentum = 0;
tr_config.weight_decay = 0.0001;
tr_config.batch_size = 50;
tr_config.validate_interval = 200;
tr_config.max_epochs = 4;
tr_config.cost_function = 'cross_entropy';
tr_config.threshold = 0.002;

input_size = size(train_input{1}(:, :, 1));
input_channel = length(train_input);
test_batch_size = 1000;


l = 1;
CNN{ l }.map_size = input_size;
CNN{ l }.output = input_channel;

l = l + 1;
CNN{ l }.type = 'convolution';
CNN{ l }.weight.shape = [5, 5];
CNN{ l }.weight.filler.type = 'xavier';
CNN{ l }.weight.learning_rate = 1;
CNN{ l }.bias.option = false;
CNN{ l }.bias.learning_rate = 2;
CNN{ l }.zero_padding.option = false;
CNN{ l }.output = 20;
CNN{ l }.bottom = '';

l = l + 1;
CNN{ l }.type = 'batch_normalization';
CNN{ l }.BN_decay = 0.99;

l = l + 1;
CNN{ l }.type = 'sampling';
CNN{ l }.sampling.type = 'max';
CNN{ l }.sampling.shape = [2, 2];
CNN{ l }.sampling.stride = [2, 2];

l = l + 1;
CNN{ l }.type = 'convolution';
CNN{ l }.weight.shape = [5, 5];
CNN{ l }.weight.filler.type = 'xavier';
CNN{ l }.weight.learning_rate = 1;
CNN{ l }.bias.option = false;
CNN{ l }.bias.learning_rate = 2;
CNN{ l }.zero_padding.option = false;
CNN{ l }.output = 40;

l = l + 1;
CNN{ l }.type = 'batch_normalization';
CNN{ l }.BN_decay = 0.99;

l = l + 1;
CNN{ l }.type = 'sampling';
CNN{ l }.sampling.type = 'max';
CNN{ l }.sampling.shape = [2, 2];
CNN{ l }.sampling.stride = [2, 2];

l = l + 1;
CNN{ l }.type = 'full_connection';
CNN{ l }.weight.filler.type = 'xavier';
CNN{ l }.weight.learning_rate = 1;
CNN{ l }.bias.learning_rate = 2;
CNN{ l }.dropout.option = true;
CNN{ l }.dropout.rate = 0.5;
CNN{ l }.output = 200;

l = l + 1;
CNN{ l }.type = 'activation';
CNN{ l }.activation = 'relu';

l = l + 1;
CNN{ l }.type = 'full_connection';
CNN{ l }.weight.filler.type = 'xavier';
CNN{ l }.weight.learning_rate = 1;
CNN{ l }.bias.learning_rate = 2;
CNN{ l }.dropout.option= false;
CNN{ l }.output = 10;

l = l + 1;
CNN{ l }.type = 'activation';
CNN{ l }.activation = 'softmax';

CNN = CNN_initialization(CNN);

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check(train_input, train_target, CNN, tr_config.cost_function, check_size, epsilon, tolerance);

tic;

CNN = CNN_train(train_input, train_target, validation_input, validation_target, tr_config, CNN);

result.train_time = toc;

[result.accuracy, result.confusion_matrix] = CNN_test(test_input, test_target, CNN, test_batch_size);

writing_log(CNN, result, tr_config);

