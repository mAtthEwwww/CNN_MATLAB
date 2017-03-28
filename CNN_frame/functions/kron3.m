function D = kron3( A , b )

D = zeros( [size( A( : , : , 1 ) ) .* size( b ) , size( A , 3 ) ] );

for n = 1 : size( A , 3 )
    D( : , : , n ) = kron( A( : , : , n ) , b );
end

end