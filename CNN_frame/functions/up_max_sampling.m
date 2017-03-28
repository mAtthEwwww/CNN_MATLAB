function Y = up_max_sampling( delta , max_position, up_map_size , sampling_size , stride , pad_size )

[ rows , columns , N ] = size( delta );

Y = zeros( [ up_map_size+pad_size , N ] );

for r = 1 : rows
    for c = 1 : columns
        x = ceil( max_position( r , c , : ) / sampling_size(2) );
        y = mod( max_position( r , c , : ) , sampling_size(2) );
        y ( y==0 ) = sampling_size(2);
        position = zeros( sampling_size(1) , sampling_size(2) , N );
        for n = 1 : N
            position( x(n) , y(n) , n ) = 1;
        end
        Y( (r-1)*stride+1 : (r-1)*stride+sampling_size(1) , (c-1)*stride+1 : (c-1)*stride+sampling_size(2) , : ) = Y( (r-1)*stride+1 : (r-1)*stride+sampling_size(1) , (c-1)*stride+1 : (c-1)*stride+sampling_size(2) , : ) + bsxfun( @times , delta( r , c , : ) , position );
    end
end

Y = Y( 1 : end-pad_size(1) , 1 : end-pad_size(2) , : );

end