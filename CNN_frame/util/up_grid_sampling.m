function Y = up_grid_sampling( delta , up_map_size , sampling_size , stride , pad_size)

[ rows , columns , N ] = size( delta );

Y = zeros([up_map_size+pad_size, N]);

for r = 1 : rows
    for c = 1 : columns
        Y((r-1)*stride(1)+1, (c-1)*stride(2)+1, : ) = Y((r-1)*stride(1)+1, (c-1)*stride(2)+1, : ) + delta( r , c , : );
    end
end

Y = Y( 1 : end-pad_size(1) , 1 : end-pad_size(2) , : );

end
