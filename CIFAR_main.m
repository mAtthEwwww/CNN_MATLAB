
%% 导入已经做好预处理的数据

rand( 'state' , 0 );                                            %设置随机数种
randn( 'state' , 0 );                                           %设置随机数种

addpath 'CIFAR_10_dataset'
addpath 'CNN_frame'
addpath 'CNN_frame/functions'
addpath 'CNN_frame/constructor'

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
input_maps = length( train_input );

% release memory
clear train
clear test


%% -------------configure the training strategy-------------
tr_config.learning_period = [ 1 ];          %按照epochs来分割各个训练阶段
tr_config.learning_rate = [ 0.0005 ];%为各个训练阶段设置学习速率

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
tr_config.max_epochs = 12;

% configure the cost function
tr_config.cost_function = 'cross_entropy';

% configure the threshold of cost value
tr_config.threshold = 0.05;

% configure the initialization method for weighted and bias
weight_filler = 'gaussian';

% configure the activation function
activation = 'relu';

%% -------------configure the structure of CNN-------------
CNN{ 1 }.output = input_maps;
CNN{ 1 }.map_size = input_size;:

CNN{ 2 }.type = 'convolution';                                  %类别为卷积层
CNN{ 2 }.weight_filler = weight_filler;                         %用上面预先设置好的weight_filler方法来初始化该层的权值
CNN{ 2 }.weight_std = 0.001;                                    %如果使用0均值高斯分布，那么设置初始化的标准差
CNN{ 2 }.kernel_size = [ 5 , 5 ];                               %设置卷积核的大小
CNN{ 2 }.expand = true;                                         %设置卷积时候是否对输入矩阵进行扩展，当打开该项时，需要使用大小为奇数的卷积核
CNN{ 2 }.weight_learning_rate = 1;                              %设置权值相对于全局学习速率的倍率
CNN{ 2 }.bias_learning_rate = 2;                                %设置偏置相对于全局学习速率的倍率
CNN{ 2 }.output = 32;                                           %设置输出的矩阵数

CNN{ 3 }.type = 'sampling';                                     %类别为采样层
CNN{ 3 }.method = 'max';                                        %设置采样方法
CNN{ 3 }.sampling_size = [ 3 , 3 ];                             %采样窗大小
CNN{ 3 }.stride = 2;                                            %采样窗移动步长

CNN{ 4 }.type = 'activation';                                   %类别为激活层
CNN{ 4 }.activation = 'relu';                                   %设置激活函数类型

CNN{ 5 }.type = 'convolution';
CNN{ 5 }.weight_filler = weight_filler;
CNN{ 5 }.weight_std = 0.01;
CNN{ 5 }.weight_learning_rate = 1;
CNN{ 5 }.bias_learning_rate = 2;
CNN{ 5 }.kernel_size = [ 5 , 5 ];
CNN{ 5 }.expand = true;
CNN{ 5 }.output = 48;

CNN{ 6 }.type = 'activation';
CNN{ 6 }.activation = 'relu';

CNN{ 7 }.type = 'sampling';
CNN{ 7 }.method = 'average';
CNN{ 7 }.sampling_size = [ 3 , 3 ];
CNN{ 7 }.stride = 2;

CNN{ 8 }.type = 'convolution';
CNN{ 8 }.weight_filler = weight_filler;
CNN{ 8 }.weight_std = 0.01;
CNN{ 8 }.weight_learning_rate = 1;
CNN{ 8 }.bias_learning_rate = 2;
CNN{ 8 }.kernel_size = [ 3 , 3 ];
CNN{ 8 }.expand = true;
CNN{ 8 }.output = 64;

CNN{ 9 }.type = 'activation';
CNN{ 9 }.activation = 'relu';

CNN{ 10 }.type = 'sampling';
CNN{ 10 }.method = 'average';
CNN{ 10 }.sampling_size = [ 3 , 3 ];
CNN{ 10 }.stride = 2;

CNN{ 11 }.type = 'full_connection';
CNN{ 11 }.weight_filler = weight_filler;
CNN{ 11 }.weight_std = 0.1;
CNN{ 11 }.weight_learning_rate = 1;
CNN{ 11 }.bias_learning_rate = 2;
CNN{ 11 }.output = 32;
CNN{ 11 }.dropout = false;                                      %dropout技术，有待实现。

CNN{ 12 }.type = 'activation';
CNN{ 12 }.activation = 'relu';

CNN{ 13 }.type = 'full_connection';
CNN{ 13 }.weight_filler = weight_filler;
CNN{ 13 }.weight_std = 0.1;
CNN{ 13 }.weight_learning_rate = 1;
CNN{ 13 }.bias_learning_rate = 2;
CNN{ 13 }.output = 10;

CNN{ 14 }.type = 'activation';                                  %类别为激活函数层，输出层可以规约为激活函数层
CNN{ 14 }.activation = 'softmax';                               %激活函数为softmax函数

%% 开始训练和测试
CNN = CNN_initialization( CNN );                                %初始化网络

                                                                %进行梯度检验
% check_size = 5;                                               %参与检验的样本的大小1
% epsilon = 1e-8;                                               %epsilon
% tolerance = 1e-7;                                             %梯度检验的容差
% CNN_gradient_check( train_input , train_target , CNN , tr_config.cost_function , check_size , epsilon , tolerance );

tic;
CNN = CNN_train( train_input , train_target , validation_input , validation_target , tr_config , CNN );            %训练
train_time = toc;
clear input
clear target

[accuracy, confusion_matrix] = CNN_test( test_input, test_target, CNN );        %测试，记录错误率以及错误分类样本的索引
result = { CNN , accuracy , confusion_matrix };                              %输出的结果，以及正确率，以及分类错误索引


%% 以下输出训练日志
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

tr_config.learning_rate = tr_config.learning_rate(1);%%%
run_str = sprintf( 'Accuracy %.2f%%   cost %f   time %.1fs   epochs %i   learning rate %.7f   batchsize %i   momentum %.1f   half life %i   activation %s   weight filler %s', ...
     accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , tr_config.momentum , tr_config.half_life , activation , weight_filler );
log_str = sprintf( '%s\r\n%s' , run_str , struct_str );                 

%将程序运行数据输出到日志
fid = fopen('Log\Log.txt','a');
fprintf( fid , '%s\r\n\r\n' , log_str );
fclose(fid);

title_str = sprintf( 'AC:%.2f%%, cost:%.4f, TIME:%.1fs, EPOCHS:%i, LR:%.7f, BATCHSIZE:%i, %s', ...
     accuracy*100 , CNN{1}.cost , train_time , CNN{1}.epochs , tr_config.learning_rate , tr_config.batch_size , activation );
%给函数图像添加文字标题
title( title_str );
ylim([0 2.5]);

%将图片按照输出到"Log\image_name.png"
saveas( gcf , sprintf( '%s%s%s' , 'Log\' , run_str , '.png' ) );

% end
