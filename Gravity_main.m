% Gravity_main.m
% CNN for Gravity Wave detection
% see more detail in CIFAR_main.m

clear
clc

rand( 'state' , 0 );
randn( 'state' , 0 );

addpath 'datasets/Gravity_Wave'
addpath 'CNN_frame'
addpath 'CNN_frame/functions'
addpath 'CNN_frame/constructor'

[ data , labels ] = prepare( 100000 , 0.5 );

data = permute(data,[3,2,1]);

train_input = data( : , : , 1:90000 );
Mean = mean( train_input , 3 );
train_input = { bsxfun( @minus , train_input , Mean ) };
train_target = labels( 1:90000 , : );

validation_input = data( : , : , 90001:95000 );
validation_input = { bsxfun( @minus , validation_input , Mean ) };
validation_target = labels( 90001:95000 , : );

test_input = data( : , : , 95001:end );
test_input = { bsxfun( @minus , test_input , Mean ) };
test_target = labels( 95001:end , : );

% train_input is a array of cell
% length(train_input) is the number of input channel
% the input channel is 1 in this problem (besides, colour image has R, G, B channel)
% size(train_input{1}) equal to (1 x d x n), where d is the length of signal, and n is the number of example
% size(train_target) equal to (n x 2),


tr_config.learning_rate = 0.03;
tr_config.half_life = 20;
tr_config.momentum = 0;
tr_config.weight_decay = 0.0001;
tr_config.batch_size = 100;
tr_config.validate_interval = 900;
tr_config.max_epochs = 80;
tr_config.cost_function = 'cross_entropy';
tr_config.threshold = 0.05;
weight_filler.type = 'gaussian';
activation = 'relu';
output_function = 'softmax';
input_channel = size(length(train_input));
input_size = size(train_input{1}(:,:,1));


CNN{ 1 }.output = input_channel;
CNN{ 1 }.map_size = input_size;

CNN{ 2 }.type = 'convolution';
CNN{ 2 }.weight_filler = weight_filler;
CNN{ 2 }.weight_filler.std = 0.1;
CNN{ 2 }.kernel_size = [ 1 , 24 ];
CNN{ 2 }.expand = false;
CNN{ 2 }.weight_learning_rate = 1;
CNN{ 2 }.bias_learning_rate = 2;
CNN{ 2 }.output = 15;

CNN{ 3 }.type = 'activation';
CNN{ 3 }.activation = 'relu';

CNN{ 4 }.type = 'sampling';
CNN{ 4 }.method = 'max';
CNN{ 4 }.sampling_size = [ 1 , 4 ];
CNN{ 4 }.stride = 4;

CNN{ 5 }.type = 'convolution';
CNN{ 5 }.weight_filler = weight_filler;
CNN{ 5 }.weight_filler.std = 0.1;
CNN{ 5 }.kernel_size = [ 1 , 12 ];
CNN{ 5 }.expand = false;
CNN{ 5 }.weight_learning_rate = 1;
CNN{ 5 }.bias_learning_rate = 2;
CNN{ 5 }.output = 20;

CNN{ 6 }.type = 'activation';
CNN{ 6 }.activation = 'relu';

CNN{ 7 }.type = 'sampling';
CNN{ 7 }.method = 'max';
CNN{ 7 }.sampling_size = [ 1 , 4 ];
CNN{ 7 }.stride = 4;

CNN{ 8 }.type = 'convolution';
CNN{ 8 }.weight_filler = weight_filler;
CNN{ 8 }.weight_filler.std = 0.1;
CNN{ 8 }.kernel_size = [ 1 , 6 ];
CNN{ 8 }.expand = false;
CNN{ 8 }.weight_learning_rate = 1;
CNN{ 8 }.bias_learning_rate = 2;
CNN{ 8 }.output = 25;

CNN{ 9 }.type = 'activation';
CNN{ 9 }.activation = 'relu';

CNN{ 10 }.type = 'sampling';
CNN{ 10 }.method = 'max';
CNN{ 10 }.sampling_size = [ 1 , 4 ];
CNN{ 10 }.stride = 4;

CNN{ 11 }.type = 'full_connection';
CNN{ 11 }.weight_filler = weight_filler;
CNN{ 11 }.weight_filler.std = 0.5;
CNN{ 11 }.weight_learning_rate = 1;
CNN{ 11 }.bias_learning_rate = 2;
CNN{ 11 }.output = 10;
CNN{ 11 }.dropout = false;

CNN{ 12 }.type = 'activation';
CNN{ 12 }.activation = 'relu';

CNN{ 13 }.type = 'full_connection';
CNN{ 13 }.weight_filler = weight_filler;
CNN{ 13 }.weight_filler.std = 0.5;
CNN{ 13 }.weight_learning_rate = 1;
CNN{ 13 }.bias_learning_rate = 2;
CNN{ 13 }.output = 2;

CNN{ 14 }.type = 'activation';
CNN{ 14 }.activation = output_function;

CNN = CNN_initialization( CNN );


% check_size = 10;
% epsilon = 1e-7;
% tolerance = 1e-7;
% CNN_gradient_check( train_input , train_target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );

tic;
CNN = CNN_train(train_input, train_target, validation_input, validation_target, tr_config , CNN);
train_time = toc;
clear input
clear target

[ accuracy , confusion_matrix ] = CNN_test( test_input, test_target, CNN );


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
s
