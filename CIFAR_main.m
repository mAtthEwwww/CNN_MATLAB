% CIFAR_main.m

clear
clc

% configure random seed
rand( 'state' , 0 );
randn( 'state' , 0 );

% add path
addpath 'CNN_frame'
addpath 'CNN_frame/util'
addpath 'CNN_frame/layer_constructor'
addpath 'CNN_frame/layer_calculator'

addpath 'datasets/CIFAR_10_dataset'

%% -------------prepare data set-------------

% Load the cifar-10 data set, with with pre-treated type 2.
% Train and test are struct, both of which contain a input set X and a output set T.
% The input set X is an array of cell (X{1}, X{2}, X{3}), each cell contains a 3-order tensor. Each tensor (height x width x sample-size) presents a color channel (RGB)
% The output set T is a label matrix, each row is a label vector of a sample (one-hot)
%[ train , test ] = load_cifar( 2 );
[train, test] = load_cifar_tiny( 2 );

% set the size of validate set
validation_size = 2000;

% the train set for training
train_input = train.X;
train_target = train.T;

% the test set for testing
test_input = test.X;
test_target = test.T;

% divide a validate set from test set
[validation_input, validation_target] = get_mini_batch( test_input , test_target , 1 : validation_size );

% configure the image size (height x width)
input_size = size(train_input{1}(:, :, 1));

% configure the number of color channel
input_channel = length(train_input);

% configure the size of a testing batch
test_batch_size = 2000;

% release memory
clear train
clear test


%% -------------configure the training strategy-------------

% configure the initial learning rate
tr_config.learning_rate = 0.0005;

% configure the strategy of learning rate decay
% the half life of learning rate (epoch)
tr_config.half_life = 5;

% configure the momentum of gradient descent
tr_config.momentum = 0.9;

% configure the parameter of weight decay
tr_config.weight_decay = 0.00001;

% configure the batch-size of stochastic gradient descent
tr_config.batch_size = 10;

% configure the interval(iteration) between successive validation
tr_config.validate_interval = 5000;

% configure the maximum number of epochs
tr_config.max_epochs = 3;

% configure the cost function
tr_config.cost_function = 'cross_entropy';

% configure the threshold of cost value
tr_config.threshold = 0.05;


%% -------------configure the structure of CNN-------------

% input layer

% configure the number of output channel
l = 1;
CNN{l}.output = input_channel;

% configure the map size
CNN{l}.map_size = input_size;

% convolution layer
l = l + 1;
% configure the layer type
CNN{l}.type = 'convolution';

% configure the weight initialization type
CNN{l}.weight.filler.type = 'gaussian';

% configure the std for the weight
CNN{l}.weight.filler.std = 0.001;

% configure the size of convolution kernel
CNN{l}.weight.shape = [5, 5];

% configure the zero padding
% when turn on zero padding, make sure the kernel size is odd
CNN{l}.zero_padding.option = true;

% configure the relative learning rate of the weight (kernel)
CNN{l}.weight.learning_rate = 1;

% turn on or turn off the bias
CNN{l}.bias.option = true;

% configure the relative learning rate of the bias (intercept)
CNN{l}.bias.learning_rate = 2;

% configure the number of channel
CNN{l}.output = 32;


%% batch normalization layer
%l = l + 1;
%% configure the batch normalization
%CNN{l}.type = 'batch_normalization';
%% configure the moving average decay rate of batch normalization
%CNN{l}.BN_decay = 0.99;


% sampling layer
l = l + 1
CNN{l}.type = 'sampling';

% configure the down sampling type
CNN{l}.sampling.type = 'max';

% configure the size of sampling window
CNN{l}.sampling.shape = [3, 3];

% configure the stride of sliding window
CNN{l}.stride = [2, 2];


% activation layer
CNN{l}.type = 'activation';

% configure the activation function
CNN{l}.activation = 'relu';


% convolution layer
l = l + 1;

CNN{l}.type = 'convolution';

CNN{l}.weight.filler.type = weight_filler;

CNN{l}.weight_filler.std = 0.01;

CNN{l}.weight.learning_rate = 1;

CNN{l}.weigth.shape = [5 ,5];

CNN{l}.bias.option = true;

CNN{l}.bias.learning_rate = 2;

CNN{l}.zero_padding.option = true;

CNN{l}.output = 48;


%% batch normalization layer
%l = l + 1;
%
%CNN{l}.type = 'batch_normalization';
%
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

CNN{l}.output = 64;


%% batch normalization layer
%l = l + 1;
%
%CNN{l}.type = 'batch_normalization';
%
%CNN{l}.BN_decay = 0.99;



% activation layer
l = l + 1;

CNN{ l }.type = 'activation';

CNN{ l }.activation = 'relu';


% sampling layer
l = l + 1;

CNN{l}.type = 'sampling';

CNN{l}.sampling.type = 'average';

CNN{l}.sampling.shape = [ 3 , 3 ];

CNN{l}.sampling.stride = 2;


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



%% -------------initializaing the CNN-------------------

CNN = CNN_initialization( CNN );


%% --------------check the gradient---------------------

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check( train_input , train_target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );


%% -----------------train the CNN-----------------------
tic;
CNN = CNN_train( train_input , train_target , validation_input , validation_target , tr_config , CNN );

train_time = toc;
clear train_input
clear train_target

%% ----------------test the CNN------------------------
[accuracy, confusion_matrix] = CNN_test( test_input, test_target, CNN, test_batch_size);



%% ------------save the running log and figure---------

% prepare the string with the information of structure
struct_str = sprintf( 'layer 1   type: input                          width:%i\r\n' , CNN{ 1 }.output );
for l = 2 : length( CNN )
    if strcmp( CNN{l}.type , 'convolution' )
        struct_str = sprintf( '%slayer %i   type: convolution    kernel size: %ix%i   width: %i\r\n' , struct_str , l , CNN{l}.kernel_size(1) , CNN{l}.kernel_size(2) , CNN{ l }.output );
    elseif strcmp( CNN{l}.type , 'sampling' )
        struct_str = sprintf( '%slayer %i   type: sampling     stride: %i   sampling size: %ix%i   sampling type: %s\r\n' , struct_str , l , CNN{l}.stride , CNN{l}.sampling_size(1) , CNN{l}.sampling_size(2) , CNN{ l }.type );
    elseif strcmp( CNN{l}.type , 'full_connection' )
        struct_str = sprintf( '%slayer %i   type: full_connection                width: %i\r\n' , struct_str , l , CNN{ l }.output );
    elseif strcmp( CNN{l}.type, 'activation' )
        struct_str = sprintf( '%slayer %i   type: activation             %s\r\n' , struct_str, l, CNN{l}.activation );
    end
end

% prepare the string with the information of running result
run_str = sprintf('Accuracy %.2f%%   cost %f   time %.1fs   epochs %i   learning rate %.7f   batchsize %i   momentum %.1f   half life %i   weight filler %s', accuracy*100, CNN{1}.cost, train_time, CNN{1}.epochs, tr_config.learning_rate, tr_config.batch_size, tr_config.momentum, tr_config.half_life, weight_filler.type);
log_str = sprintf('%s\r\n%s', run_str, struct_str);                 

% output the string to /Log/Log
fid = fopen('Log/Log', 'a');
fprintf(fid, '%s\r\n\r\n', log_str);
fclose(fid);

% prepare the title string
title_str = sprintf( 'AC:%.2f%%, cost:%.4f, TIME:%.1fs, EPOCHS:%i, LR:%.7f, BATCHSIZE:%i', accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size);

% add title to the figure
title( title_str );

% configure the y interval of figure
ylim([0 2.5]);

% output the figure to
saveas( gcf , sprintf( '%s%s%s' , 'Log/' , run_str , '.png' ) );
