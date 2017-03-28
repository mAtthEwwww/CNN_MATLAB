% 详细注释见cifar_CNN.m

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
tr_config.learning_period = [ 1 ];
tr_config.half_life = 5;
tr_config.momentum = 0;
tr_config.weight_decay = 0.0001;
tr_config.batch_size = 50;
tr_config.validate_interval = 200;
tr_config.max_epochs = 20;
tr_config.cost_function = 'cross_entropy';
tr_config.threshold = 0.002;

weight_filler = 'xavier';
activation = 'relu';        %only for log


CNN{ 1 }.output = 1;
CNN{ 1 }.map_size = size( input{ 1 }( : , : , 1 ) );

CNN{ 2 }.type = 'convolution';
CNN{ 2 }.weight_filler = weight_filler;
CNN{ 2 }.weight_std = 0.0001;
CNN{ 2 }.weight_learning_rate = 1;
CNN{ 2 }.bias_learning_rate = 2;
CNN{ 2 }.output = 20;
CNN{ 2 }.kernel_size = [5, 5];
CNN{ 2 }.expand = false;

CNN{ 3 }.type = 'sampling';                            %第三层属性为采样层
CNN{ 3 }.method = 'max';
CNN{ 3 }.sampling_size = [2, 2];                        %第三层采样窗直径 map size: 24/2=129
CNN{ 3 }.stride = 2;

CNN{ 4 }.type = 'convolution';
CNN{ 4 }.weight_filler = weight_filler;
CNN{ 4 }.weight_std = 0.001;                 %仅当weight_filler为gaussian时有效
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
CNN{ 6 }.weight_std = 0.01;              %仅当weight_filler为gaussian时有效
CNN{ 6 }.weight_learning_rate = 1;
CNN{ 6 }.bias_learning_rate = 2;
CNN{ 6 }.output = 200;

CNN{ 7 }.type = 'activation';
CNN{ 7 }.activation = 'relu';

CNN{ 8 }.type = 'full_connection';                            %属性为全连接层
CNN{ 8 }.weight_filler = 'xavier';
CNN{ 8 }.weight_std = 0.1;               %仅当weight_filler为gaussian时有效
CNN{ 8 }.weight_learning_rate = 1;
CNN{ 8 }.bias_learning_rate = 2;
CNN{ 8 }.output = 10;

CNN{ 9 }.type = 'activation';
CNN{ 9 }.activation = 'softmax';

CNN = CNN_initialization( CNN );

% check_size = 5;
% epsilon = 1e-8;
% tolerance = 1e-7;
% CNN_gradient_check( input , target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );

tic;
CNN = CNN_train( train_input, train_target, validation_input, validation_target, tr_config, CNN );
train_time = toc;

[error, bad] = CNN_test( test_input, test_target, CNN );
accuracy = 1 - error;
result = { accuracy , CNN , bad };


struct_str = sprintf( 'layer 1   type:input                            width:%i\r\n' , CNN{ 1 }.output );
for l = 2 : length( CNN )
    if strcmp( CNN{l}.type , 'convolution' )
        struct_str = sprintf( '%slayer %i   type: convolution    kernel size: %i   width: %i\r\n' , struct_str , l , CNN{l}.kernel_size , CNN{ l }.output );
    elseif strcmp( CNN{l}.type , 'sampling' )
        struct_str = sprintf( '%slayer %i   type: sampling     sampling size: %i   method: %s\r\n' , struct_str , l , CNN{l}.sampling_size , CNN{ l }.method );
    elseif strcmp( CNN{l}.type , 'full_connection' )
        struct_str = sprintf( '%slayer %i   type: full_connection                 width: %i\r\n' , struct_str , l , CNN{ l }.output );
    elseif strcmp( CNN{l}.type, 'activation' )
        struct_str = sprintf( '%slayer %i   type: activation             %s\r\n' , struct_str, l, CNN{l}.activation );
    end
end

run_str = sprintf( 'Accuracy %.2f%%   cost %f   time %.1fs   epochs %i   learning rate %.5f   batchsize %i   momentum %.1f   half life %i   activation %s   weight filler %s', ...
     accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , tr_config.momentum , tr_config.half_life , activation , weight_filler );
log_str = sprintf( '%s\r\n%s' , run_str , struct_str );                 

%将程序运行数据输出到日志
fid = fopen('Log\Log.txt','a');
fprintf( fid , '%s\r\n\r\n' , log_str );
fclose(fid);

title_str = sprintf( 'AC:%.2f%%, cost:%.4f, TIME:%.1fs, EPOCHS:%i, LR:%.5f, BATCHSIZE:%i, %s', ...
     accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , activation );
%给函数图像添加文字标题
title( title_str );
ylim([0 1]);

%将图片按照输出到"Log\image_name.png"
saveas( gcf , sprintf( '%s%s%s' , 'Log\' , run_str , '.png' ) );
