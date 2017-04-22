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
[ train , test ] = load_MNIST_2d(3);
train_input = {train.X};
train_target = train.T;
test_input = {test.X};
test_target = test.T;
validation_size = 1000;
[validation_input, validation_target] = get_mini_batch(test_input, test_target, 1 : validation_size);
clear train
clear test

tr_config.learning_rate = 0.01;
tr_config.half_life = 3;
tr_config.momentum = 0;
tr_config.weight_decay = 0.0001;
tr_config.batch_size = 50;
tr_config.validate_interval = 200;
tr_config.max_epochs = 10;
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
CNN{ l }.bias.option = true;
CNN{ l }.bias.learning_rate = 2;
CNN{ l }.zero_padding.option = false;
%CNN{ l }.output = 20;
CNN{ l }.output = 3;
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
CNN{ l }.bias.option = true;
CNN{ l }.bias.learning_rate = 2;
CNN{ l }.zero_padding.option = false;
%CNN{ l }.output = 40;
CNN{ l }.output = 5;

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
CNN{ l }.dropout.option = false;
CNN{ l }.dropout.rate = 0.5;
%CNN{ l }.output = 200;
CNN{ l }.output = 12;

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

check_size = 10;
epsilon = 1e-6;
tolerance = 1e-8;
CNN_gradient_check( train_input , train_target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );

tic;
CNN = CNN_train( train_input, train_target, validation_input, validation_target, tr_config, CNN );
train_time = toc;

[accuracy, confusion_matrix] = CNN_test( test_input, test_target, CNN, test_batch_size );

struct_str = sprintf( 'layer 1   type: input                          width:%i\r\n' , CNN{ 1 }.output );
for l = 2 : length( CNN )
    if strcmp( CNN{l}.type , 'convolution' )
        struct_str = sprintf( '%slayer %i   type: convolution    kernel size: %ix%i   width: %i\r\n', struct_str, l, CNN{l}.weight.shape(1), CNN{l}.weight.shape(2), CNN{l}.output);
    elseif strcmp( CNN{l}.type , 'sampling' )
        struct_str = sprintf( '%slayer %i   type: sampling     stride: %ix%i   sampling size: %ix%i   sampling type: %s\r\n', struct_str, l, CNN{l}.sampling.stride(1), CNN{l}.sampling.stride(2), CNN{l}.sampling.shape(1), CNN{l}.sampling.shape(2), CNN{ l }.sampling.type);
    elseif strcmp( CNN{l}.type , 'full_connection' )
        struct_str = sprintf( '%slayer %i   type: full_connection                width: %i\r\n' , struct_str , l , CNN{ l }.output );
    elseif strcmp( CNN{l}.type, 'activation' )
        struct_str = sprintf( '%slayer %i   type: activation             %s\r\n' , struct_str, l, CNN{l}.activation );
    end
end
run_str = sprintf('Accuracy %.2f%%   cost %f   time %.1fs   epochs %i   learning rate %.7f   batchsize %i   momentum %.1f   half life %i   activation %s   weight filler %s', accuracy * 100, CNN{1}.cost, train_time, CNN{1}.epochs, tr_config.learning_rate, tr_config.batch_size, tr_config.momentum, tr_config.half_life, activation, weight_filler.type);
log_str = sprintf( '%s\r\n%s' , run_str , struct_str );                 

% output the running log
fid = fopen('Log/Log', 'a');
fprintf(fid, '%s\r\n\r\n', log_str );
fclose(fid);

title_str = sprintf( 'AC:%.2f%%, cost:%.4f, TIME:%.1fs, EPOCHS:%i, LR:%.7f, BATCHSIZE:%i', ...
     accuracy*100, CNN{1}.cost, train_time, CNN{1}.epochs, tr_config.learning_rate, tr_config.batch_size);

title( title_str );
ylim([0 2.5]);

% output the figure
saveas( gcf , sprintf( '%s%s%s' , 'Log/' , run_str , '.png' ) );
