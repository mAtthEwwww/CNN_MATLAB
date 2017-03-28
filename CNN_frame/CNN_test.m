function [ accuracy , confusion_matrix ] = CNN_test( input, target, CNN )

K = size( target , 2 );

CNN = CNN_feedforward( input, CNN );

[ ~ , target ] = max( target , [] , 2 ); 

[ ~ , T_hat ] = max( CNN{ length( CNN ) }.X , [] , 2 );

accuracy = mean( target == T_hat );

confusion_matrix = zeros( K );

for r = 1 : K
    for c = 1 : K
        confusion_matrix( r , c ) =  sum( T_hat( target == r ) == c );
    end
end

end
        

% T_hat = regularize( CNN{ length( CNN ) }.X );
% 
% bad = max( xor( T_hat, target ), [], 2 );
% 
% error = sum( bad ) / length( bad );