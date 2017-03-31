% CIFAR_main.m

clear
clc

% configure random seed
rand( 'state' , 0 );
randn( 'state' , 0 );

% add path
addpath 'datasets/CIFAR_10_dataset'
addpath 'CNN_frame'
addpath 'CNN_frame/functions'
addpath 'CNN_frame/constructor'

%% -------------prepare data set-------------

% Load the cifar-10 data set, with with pre-treated method 2.
% Train and test are struct, both of which contain a input set X and a output set T.
% The input set X is an array of cell (X{1}, X{2}, X{3}), each cell contains a 3-order tensor. Each tensor (height x width x sample-size) presents a color channel (RGB)
% The output set T is a label matrix, each row is a label vector of a sample (one-hot)
[ train , test ] = load_cifar( 2 );

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
input_size = size( train_input{1}( : , : , 1 ) );

% configure the number of color channel
input_channel = length( train_input );

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

% configure the initialization method for weighted and bias
weight_filler.type = 'gaussian';

% configure the activation function
activation = 'relu';

% configure the output function
output_function = 'softmax';

%% -------------configure the structure of CNN-------------

% input layer

% configure the number of output channel
CNN{ 1 }.output = input_channel;

% configure the map size
CNN{ 1 }.map_size = input_size;

% convolution layer
% configure the layer type
CNN{ 2 }.type = 'convolution';

% configure the weight initialization method
CNN{ 2 }.weight_filler = weight_filler;

% configure the std for the weight
CNN{ 2 }.weight_filler.std = 0.001;

% configure the size of convolution kernel
CNN{ 2 }.kernel_size = [ 5 , 5 ];

% configure the zero padding
% when turn on zero padding, make sure the kernel size is odd
CNN{ 2 }.expand = true;

% configure the relative learning rate of the weight (kernel)
CNN{ 2 }.weight_learning_rate = 1;

% configure the relative learning rate of the bias (intercept)
CNN{ 2 }.bias_learning_rate = 2;

% configure the number of channel
CNN{ 2 }.output = 32;


% sampling layer

CNN{ 3 }.type = 'sampling';

% configure the down sampling method
CNN{ 3 }.method = 'max';

% configure the size of sampling window
CNN{ 3 }.sampling_size = [3, 3];

% configure the stride of sliding window
CNN{ 3 }.stride = 2;


% activation layer
CNN{ 4 }.type = 'activation';

% configure the activation function
CNN{ 4 }.activation = activation;


% convolution layer

CNN{ 5 }.type = 'convolution';

CNN{ 5 }.weight_filler = weight_filler;

CNN{ 5 }.weight_filler.std = 0.01;

CNN{ 5 }.weight_learning_rate = 1;

CNN{ 5 }.bias_learning_rate = 2;

CNN{ 5 }.kernel_size = [5 ,5];

CNN{ 5 }.expand = true;

CNN{ 5 }.output = 48;

% activation layer

CNN{ 6 }.type = 'activation';

CNN{ 6 }.activation = activation;


% sampling layer

CNN{ 7 }.type = 'sampling';

CNN{ 7 }.method = 'average';

CNN{ 7 }.sampling_size = [ 3 , 3 ];

CNN{ 7 }.stride = 2;


% convolution layer

CNN{ 8 }.type = 'convolution';

CNN{ 8 }.weight_filler = weight_filler;

CNN{ 8 }.weight_filler.std = 0.01;

CNN{ 8 }.weight_learning_rate = 1;

CNN{ 8 }.bias_learning_rate = 2;

CNN{ 8 }.kernel_size = [ 3 , 3 ];

CNN{ 8 }.expand = true;

CNN{ 8 }.output = 64;


% activation layer

CNN{ 9 }.type = 'activation';

CNN{ 9 }.activation = 'relu';


% sampling layer

CNN{ 10 }.type = 'sampling';

CNN{ 10 }.method = 'average';

CNN{ 10 }.sampling_size = [ 3 , 3 ];

CNN{ 10 }.stride = 2;


% full connection layer

CNN{ 11 }.type = 'full_connection';

CNN{ 11 }.weight_filler = weight_filler;

CNN{ 11 }.weight_filler.std = 0.1;

CNN{ 11 }.weight_learning_rate = 1;

CNN{ 11 }.bias_learning_rate = 2;

CNN{ 11 }.output = 32;

% the function of dropout is not complete
CNN{ 11 }.dropout = false;


% activation layer

CNN{ 12 }.type = 'activation';

CNN{ 12 }.activation = 'relu';


% full connection layer

CNN{ 13 }.type = 'full_connection';

CNN{ 13 }.weight_filler = weight_filler;

CNN{ 13 }.weight_filler.std = 0.1;

CNN{ 13 }.weight_learning_rate = 1;

CNN{ 13 }.bias_learning_rate = 2;

CNN{ 13 }.output = 10;


% output layer

CNN{ 14 }.type = 'activation';

CNN{ 14 }.activation = output_function;



%% -------------initializaing the CNN-------------------

CNN = CNN_initialization( CNN );


%% --------------check the gradient---------------------

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check( train_input , train_target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );


%% -----------------train the CNN-----------------------
tic;
CNN = CNN_train( train_input , train_target , validation_input , validation_target , tr_config , CNN );            %ÑµÁ·

train_time = toc;
clear train_input
clear train_target

%% ----------------test the CNN------------------------
[accuracy, confusion_matrix] = CNN_test( test_input, test_target, CNN );



%% ------------save the running log and figure---------

% prepare the string with the information of structure
struct_str = sprintf( 'layer 1   type: input                          width:%i\r\n' , CNN{ 1 }.output );
for l = 2 : length( CNN )
    if strcmp( CNN{l}.type , 'convolution' )
        struct_str = sprintf( '%slayer %i   type: convolution    kernel size: %ix%i   width: %i\r\n' , struct_str , l , CNN{l}.kernel_size(1) , CNN{l}.kernel_size(2) , CNN{ l }.output );
    elseif strcmp( CNN{l}.type , 'sampling' )
        struct_str = sprintf( '%slayer %i   type: sampling     stride: %i   sampling size: %ix%i   method: %s\r\n' , struct_str , l , CNN{l}.stride , CNN{l}.sampling_size(1) , CNN{l}.sampling_size(2) , CNN{ l }.method );
    elseif strcmp( CNN{l}.type , 'full_connection' )
        struct_str = sprintf( '%slayer %i   type: full_connection                width: %i\r\n' , struct_str , l , CNN{ l }.output );
    elseif strcmp( CNN{l}.type, 'activation' )
        struct_str = sprintf( '%slayer %i   type: activation             %s\r\n' , struct_str, l, CNN{l}.activation );
    end
end

% prepare the string with the information of running result
run_str = sprintf( 'Accuracy %.2f%%   cost %f   time %.1fs   epochs %i   learning rate %.7f   batchsize %i   momentum %.1f   half life %i   activation %s   weight filler %s', accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , tr_config.momentum , tr_config.half_life , activation , weight_filler.type );
log_str = sprintf( '%s\r\n%s' , run_str , struct_str );                 

% output the string to /Log/Log
fid = fopen('Log/Log', 'a');
fprintf(fid, '%s\r\n\r\n', log_str );
fclose(fid);

% prepare the title string
title_str = sprintf( 'AC:%.2f%%, cost:%.4f, TIME:%.1fs, EPOCHS:%i, LR:%.7f, BATCHSIZE:%i, %s', accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , activation );

% add title to the figure
title( title_str );

% configure the y interval of figure
ylim([0 2.5]);

% output the figure to
saveas( gcf , sprintf( '%s%s%s' , 'Log/' , run_str , '.png' ) );
