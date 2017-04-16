function [ accuracy , confusion_matrix ] = CNN_test( input, target, CNN , batch_size )
% CNN_test.m
% calculate the CNN performance on (input, target) set
%
% Inputs:
%       input  is a array of cell, each cell is a 3-order tensor,
%                which represents a color channel
%       target  is a bool matrix, each row represents a one-hot label of a sample
% 
% Output:
%       accuracy  is the accuracy of classification
%       confusion_matrix  is a square matrix, called confusion matrix


% extract the number of class
[N, K] = size(target);

% extract the class labels of target set
[ ~ , target ] = max(target , [], 2); 

% construct an empty vector for inference labels
T_hat = zeros(size(target));

if nargin == 3
    batch_size = length(target);
end

for i = 1 : floor(N / batch_size)
    batch_input = get_mini_batch(input, (i-1) * batch_size + 1 : min(i * batch_size, length(target)));
    CNN = CNN_feedforward(CNN, batch_input);
    [~, T_hat((i-1) * batch_size + 1 : min(i * batch_size, length(target)))] = max(CNN{length(CNN)}.X, [], 2);
end

% calculate the prediction accuracy
accuracy = mean( target == T_hat );

% construct an empty matrix for confusion matrix
confusion_matrix = zeros( K );

% fill the confusion matrix row by row, column by column
for r = 1 : K
    for c = 1 : K
        confusion_matrix(r, c) =  sum(T_hat(target == r ) == c);
    end
end

end
        
% T_hat = regularize( CNN{ length( CNN ) }.X );
% 
% bad = max( xor( T_hat, target ), [], 2 );
% 
% error = sum( bad ) / length( bad );
%
% accuracy = 1 - error;
