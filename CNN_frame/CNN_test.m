function [ accuracy , confusion_matrix ] = CNN_test( input, target, CNN )
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
K = size( target , 2 );

% feedforward with input set
CNN = CNN_feedforward(input, CNN);

% extract the class labels of target set
[ ~ , target ] = max(target , [], 2); 

% extract the prediction labels of output
[ ~ , T_hat ] = max(CNN{length(CNN)}.X, [], 2);

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
