function [ train , test ] = load_cifar( method )
load data_batch_1.mat
data_1 = data;
labels_1 = labels;
load data_batch_2.mat
data_2 = data;
labels_2 = labels;
load data_batch_3.mat
data_3 = data;
labels_3 = labels;
load data_batch_4.mat
data_4 = data;
labels_4 = labels;
load data_batch_5.mat
data_5 = data;
labels_5 = labels;
train_data = double([ data_1; data_2; data_3; data_4; data_5 ]);
train_labels = [ labels_1; labels_2; labels_3; labels_4; labels_5 ];
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
