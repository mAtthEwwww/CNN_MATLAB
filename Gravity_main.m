% Gravity_main.m
% CNN for Gravity Wave detection
% see more detail in CIFAR_main.m

clear
clc

rand( 'state' , 0 );
randn( 'state' , 0 );

addpath 'CNN_frame'
addpath 'CNN_frame/util'
addpath 'CNN_frame/layer_constructor'
addpath 'CNN_frame/layer_calculator'
addpath 'datasets/Gravity_Wave'


%% -------------------prepare data set----------------------

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

test_batch_size = 10000;

% train_input is a array of cell
% length(train_input) is the number of input channel
% the input channel is 1 in this problem (besides, colour image has R, G, B channel)
% size(train_input{1}) equal to (1 x d x n), where d is the length of signal, and n is the number of example
% size(train_target) equal to (n x 2),


%% -------------configure the training strategy-------------

tr_config.learning_rate = 0.03;
tr_config.half_life = 20;
tr_config.momentum = 0;
tr_config.weight_decay = 0.0001;
tr_config.batch_size = 100;
tr_config.validate_interval = 900;
tr_config.max_epochs = 80;
tr_config.cost_function = 'cross_entropy';
tr_config.threshold = 0.05;
input_channel = length(train_input);
input_size = size(train_input{1}(:,:,1));


l = 1;
CNN{l}.output = input_channel;
CNN{l}.map_size = input_size;

l = l + 1;
CNN{l}.type = 'convolution';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.1;
CNN{l}.weight.shape= [1, 24];
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.zero_padding.option = false;
CNN{l}.output = 15;

l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';

l = l + 1;
CNN{l}.type = 'sampling';
CNN{l}.sampling.type= 'max';
CNN{l}.sampling.shape= [1, 4];
CNN{l}.sampling.stride = [1, 4];

l = l + 1;
CNN{l}.type = 'convolution';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.1;
CNN{l}.weight.shape = [1, 12];
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.zero_padding.option = false;
CNN{l}.output = 20;

l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';

l = l + 1;
CNN{l}.type = 'sampling';
CNN{l}.sampling.type = 'max';
CNN{l}.sampling.shape = [1, 4];
CNN{l}.sampling.stride = [1, 4];

l = l + 1;
CNN{l}.type = 'convolution';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.1;
CNN{l}.weight.shape= [1, 6];
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.zero_padding.option = false;
CNN{l}.output = 25;

l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';

l = l + 1;
CNN{l}.type = 'sampling';
CNN{l}.sampling.type = 'max';
CNN{l}.sampling.shape = [1, 4];
CNN{l}.sampling.stride = [1, 4];

l = l + 1;
CNN{l}.type = 'full_connection';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.5;
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.dropout.option = true;
CNN{l}.dropout.rate = 0.5;
CNN{l}.output = 10;

l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'relu';

l = l + 1;
CNN{l}.type = 'full_connection';
CNN{l}.weight.filler.type = 'gaussian';
CNN{l}.weight.filler.std = 0.5;
CNN{l}.weight.learning_rate = 1;
CNN{l}.bias.option = true;
CNN{l}.bias.learning_rate = 2;
CNN{l}.dropout.option = false;
CNN{l}.output = 2;

l = l + 1;
CNN{l}.type = 'activation';
CNN{l}.activation = 'softmax';


%% -------------initializaing the CNN-------------------

CNN = CNN_initialization(CNN);


%% --------------check the gradient---------------------

% check_size = 10;
% epsilon = 1e-7;
% tolerance = 1e-7;
% CNN_gradient_check(train_input, train_target, CNN, tr_config.cost_function, check_size, epsilon, tolerance);


%% -----------------train the CNN-----------------------

tic;

CNN = CNN_train(train_input, train_target, validation_input, validation_target, tr_config , CNN);

train_time = toc;


%% ----------------test the CNN------------------------

[result.accuracy, result.confusion_matrix] = CNN_test(test_input, test_target, CNN, test_batch_size);


%% ------------save the running log and figure---------

writing_log(CNN, result, tr_config);

