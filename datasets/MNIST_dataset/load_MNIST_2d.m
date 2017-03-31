function [ train , test ] = load_MNIST_2d( method )

% 将本文件和一下文件放在一起
% origin_image.m, 
% loadMNISTLabels.m, 
% train-images-idx3-ubyte, 
% train-labels-idx1-ubyte,
% t10k-images-idx3-ubyte,
% t10k-labels-idx1-ubyte

% method == 1 减均值，除以标准差
% method == 2 减均值
% method == 3 除以255

train.X = loadMNISTImages_2d('train-images-idx3-ubyte');
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
test.X = loadMNISTImages_2d('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

if method == 1 || 2
    Mean = mean( train.X ,3 );
    train.X = bsxfun( @minus , train.X , Mean );
    test.X = bsxfun( @minus , test.X , Mean );
    if method == 1
        Std = std( train.X , [] , 3 );    
        train.X = bsxfun( @rdivide , train.X , Std+eps );
        test.X = bsxfun( @rdivide , test.X , Std+eps );
    end
elseif method == 3
    train.X = train.X / 255;
    test.X = test / 255;
end
    
digits = 0 : 9;

train.T = zeros( size( train_labels , 1 ) , 10 );
for k = 1 : 10
    train.T( train_labels==digits(k) , k ) = 1;
end

test.T = zeros( size( test_labels , 1 ) , 10 );
for k = 1 : 10
    test.T( test_labels==digits(k) , k ) = 1;
end

end