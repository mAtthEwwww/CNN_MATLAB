% see more detail in CIFAR_main.m

rand( 'state' , 0 );
randn( 'state' , 0 );

addpath 'MNIST_dataset'
addpath 'CNN_frame'
addpath 'CNN_frame/functions'
addpath 'CNN_frame/constructor'

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
tr_config.max_epochs = 1;
tr_config.cost_function = 'cross_entropy';
tr_config.threshold = 0.002;

input_channel = size(train_input{1}(:, :, 1));
weight_filler.type = 'xavier';
activation = 'relu';
output_function = 'softmax';


CNN{ 1 }.output = 1;
CNN{ 1 }.map_size = input_channel;

CNN{ 2 }.type = 'convolution';
CNN{ 2 }.weight_filler = weight_filler;
CNN{ 2 }.weight_learning_rate = 1;
CNN{ 2 }.bias_learning_rate = 2;
CNN{ 2 }.output = 20;
CNN{ 2 }.kernel_size = [5, 5];
CNN{ 2 }.expand = false;

CNN{ 3 }.type = 'sampling';
CNN{ 3 }.method = 'max';
CNN{ 3 }.sampling_size = [2, 2];
CNN{ 3 }.stride = 2;

CNN{ 4 }.type = 'convolution';
CNN{ 4 }.weight_filler = weight_filler;
CNN{ 4 }.weight_learning_rate = 1;
CNN{ 4 }.bias_learning_rate = 2;
CNN{ 4 }.output = 40;
CNN{ 4 }.kernel_size = [5, 5];
CNN{ 4 }.expand = false;

CNN{ 5 }.type = 'sampling';
CNN{ 5 }.method = 'max';
CNN{ 5 }.sampling_size = [2, 2];
CNN{ 5 }.stride = 2;

CNN{ 6 }.type = 'full_connection';
CNN{ 6 }.weight_filler = weight_filler;
CNN{ 6 }.weight_learning_rate = 1;
CNN{ 6 }.bias_learning_rate = 2;
CNN{ 6 }.output = 200;

CNN{ 7 }.type = 'activation';
CNN{ 7 }.activation = activation;

CNN{ 8 }.type = 'full_connection';
CNN{ 8 }.weight_filler = weight_filler;
CNN{ 8 }.weight_learning_rate = 1;
CNN{ 8 }.bias_learning_rate = 2;
CNN{ 8 }.output = 10;

CNN{ 9 }.type = 'activation';
CNN{ 9 }.activation = output_function;

CNN = CNN_initialization( CNN );

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check( train_input , train_target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );

tic;
CNN = CNN_train( train_input, train_target, validation_input, validation_target, tr_config, CNN );
train_time = toc;

[accuracy, confusion_matrix] = CNN_test( test_input, test_target, CNN );

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
run_str = sprintf( 'Accuracy %.2f%%   cost %f   time %.1fs   epochs %i   learning rate %.7f   batchsize %i   momentum %.1f   half life %i   activation %s   weight filler %s', accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , tr_config.momentum , tr_config.half_life , activation , weight_filler.type );
log_str = sprintf( '%s\r\n%s' , run_str , struct_str );                 

% output the running log
fid = fopen('Log/Log', 'a');
fprintf(fid, '%s\r\n\r\n', log_str );
fclose(fid);

title_str = sprintf( 'AC:%.2f%%, cost:%.4f, TIME:%.1fs, EPOCHS:%i, LR:%.7f, BATCHSIZE:%i, %s', ...
     accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , activation );

title( title_str );
ylim([0 2.5]);

% output the figure
saveas( gcf , sprintf( '%s%s%s' , 'Log/' , run_str , '.png' ) );
