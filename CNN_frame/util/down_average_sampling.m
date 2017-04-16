function  sample = down_average_sampling( X , map_size , sampling_size , stride , pad_size )

X( end+1 : end + pad_size(1) , end+1 : end+pad_size(2) , : ) = 0;

[ ~ , ~ , N ] = size( X );

rows = map_size(1);

columns = map_size(2);

s = rows * columns;

sample = zeros( rows * columns , prod( sampling_size ) , N );

k = 0;

for r = 1 : sampling_size(1)
    for c = 1 : sampling_size(2)
        k = k + 1;
        sample( : , k , : ) = reshape( X( r : stride(1) : end-sampling_size(1)+r , c : stride(2) : end-sampling_size(2)+c , : ) , s , 1 , N );

%        sample( : , k , : ) = reshape( X( r : stride : end-sampling_size(1)+r , c : stride : end-sampling_size(2)+c , : ) , s , 1 , N );
    end
end

sample = mean( sample , 2 );

sample = reshape( sample , rows , columns , N );

end
