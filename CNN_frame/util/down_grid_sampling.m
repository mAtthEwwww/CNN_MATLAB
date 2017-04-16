function sample = down_gird_sampling( X , map_size , sampling_size , stride , pad_size )

%X(end+1 : end + pad_size(1), end+1 : end+pad_size(2), : ) = 0;

%sample = X(1 : stride(1) : end-sampling_size(1)+1, 1 : stride(2) : end-sampling_size(2)+1, : );

sample = X(1 : stride(1) : end, 1 : stride(2) : end, : );

end
