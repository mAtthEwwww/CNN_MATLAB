function [ train , test ] = load_cifar_tiny( method )
load data_batch_1.mat

train_data = double(data);

train_labels = labels;

load test_batch.mat

test_data = double(data);

test_labels = labels;

if method == 1 || method == 2
    Mean = mean( train_data, 1 );
    train_data = bsxfun( @minus, train_data, Mean );
    test_data = bsxfun( @minus, test_data, Mean );
    if method == 1
        Std = std( train_data, [], 1 );
        train_data = bsxfun( @rdivide, train_data, Std+eps );
        test_data = bsxfun( @rdivide, test_data, Std+eps );        
    end
elseif method == 3
    train_data = train_data / 255;
    test_data = test_data / 255;
end
    
images_Red = permute( reshape( train_data( : , 1 : 1024 )', 32, 32 ,size(train_data,1) ), [ 2, 1, 3 ] );
images_Blue = permute( reshape( train_data( : , 1025 : 2048 )', 32, 32 ,size(train_data,1) ), [ 2, 1, 3 ] );
images_Green = permute( reshape( train_data( : , 2049 : 3072 )', 32, 32 ,size(train_data,1) ), [ 2, 1, 3 ] );
train.X = { images_Red, images_Blue, images_Green };

images_Red = permute( reshape( test_data( : , 1 : 1024 )', 32, 32 ,size(test_data,1) ), [ 2, 1, 3 ] );
images_Blue = permute( reshape( test_data( : , 1025 : 2048 )', 32, 32 ,size(test_data,1) ), [ 2, 1, 3 ] );
images_Green = permute( reshape( test_data( : , 2049 : 3072 )', 32, 32 ,size(test_data,1) ), [ 2, 1, 3 ] );
test.X = { images_Red, images_Blue, images_Green };

class = 0 : 9;
train.T = zeros( length( train_labels ), 10 );
for k = 1 : 10
    train.T( train_labels == class( k ), k ) = 1;
end

test.T = zeros( length( test_labels ), 10 );
for k = 1 : 10
    test.T( test_labels == class( k ), k ) = 1;
end

end
