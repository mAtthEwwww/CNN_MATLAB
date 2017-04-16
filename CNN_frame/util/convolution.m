function Y = convolution( X , kernel , pad_size )

[ r , c , n ] = size( X );

Y = zeros( r + 2*pad_size(1) , c + 2*pad_size(2) , n );

Y(pad_size(1)+1 : end-pad_size(1), pad_size(2)+1 : end-pad_size(2), : ) = X;

Y = convn( Y , kernel , 'valid' );

end
