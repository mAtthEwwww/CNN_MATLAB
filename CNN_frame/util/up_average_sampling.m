function Y = up_average_sampling( delta , up_map_size , sampling_size , stride , pad_size )

[ rows , columns , N ] = size( delta );

Y = zeros( [ up_map_size+pad_size , N ] );

for r = 1 : rows
    for c = 1 : columns
        position = ones( sampling_size(1) , sampling_size(2) , N ) / prod( sampling_size );
        Y( (r-1)*stride(1)+1 : (r-1)*stride(1)+sampling_size(1) , (c-1)*stride(2)+1 : (c-1)*stride(2)+sampling_size(2) , : ) = Y( (r-1)*stride(1)+1 : (r-1)*stride(1)+sampling_size(1) , (c-1)*stride(2)+1 : (c-1)*stride(2)+sampling_size(2) , : ) + bsxfun( @times , delta( r , c , : ) , position );

%        Y( (r-1)*stride+1 : (r-1)*stride+sampling_size(1) , (c-1)*stride+1 : (c-1)*stride+sampling_size(2) , : ) = Y( (r-1)*stride+1 : (r-1)*stride+sampling_size(1) , (c-1)*stride+1 : (c-1)*stride+sampling_size(2) , : ) + bsxfun( @times , delta( r , c , : ) , position );
    end
end

Y = Y( 1 : end-pad_size(1) , 1 : end-pad_size(2) , : );

end
