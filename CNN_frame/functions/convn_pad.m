function Y = convn_pad( X , kernel , pad )

[ r , c , n ] = size( X );

Y = zeros( r + 2*pad(1) , c + 2*pad(2) , n );

Y( pad(1)+1 : end-pad(1) , pad(2)+1 : end-pad(2) , : ) = X;

Y = convn( Y , kernel , 'valid' );

end