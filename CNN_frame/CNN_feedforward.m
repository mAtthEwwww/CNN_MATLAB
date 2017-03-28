function CNN = CNN_feedforward( X_input , CNN )

[ ~ , ~ , N ] = size( X_input{1} );

for j = 1 : CNN{1}.output
    CNN{1}.X{j} = X_input{j};
end

for l = 2 : length( CNN )
    if strcmp( CNN{l}.type, 'convolution' )
        if CNN{l}.expand == true
            CNN{l}.pad = ( CNN{l}.kernel_size - 1 ) / 2;%%%%%%%%%%%%ÐÞ¸ÄÎ»ÖÃ
            for j = 1 : CNN{l}.output
                CNN{l}.X{j} = zeros( [ CNN{l}.map_size , N ] ) + CNN{l}.bias{j};
                for i = 1 : CNN{l-1}.output
                    CNN{l}.X{j} = CNN{l}.X{j} + convn_pad( CNN{l-1}.X{i}, CNN{l}.kernel{i}{j} , CNN{l}.pad );
                end
            end
        else
            for j = 1 : CNN{l}.output
                CNN{l}.X{j} = zeros( [ CNN{l}.map_size , N ] ) + CNN{l}.bias{j};
                for i = 1 : CNN{l-1}.output
                    CNN{l}.X{j} = CNN{l}.X{j} + convn( CNN{l-1}.X{i}, CNN{l}.kernel{i}{j}, 'valid' );
                end
            end
        end
    elseif strcmp( CNN{l}.type, 'sampling' )
        for j = 1 : CNN{l}.output
            if strcmp( CNN{l}.method, 'max' )
                [ CNN{l}.X{j} , CNN{l}.max_position{j} ] = down_max_sampling( CNN{l-1}.X{j}, CNN{l}.map_size, CNN{l}.sampling_size, CNN{l}.stride, CNN{l}.pad_size );
            elseif strcmp( CNN{l}.method, 'average' )
                CNN{l}.X{j} = down_average_sampling( CNN{l-1}.X{j} , CNN{l}.map_size , CNN{l}.sampling_size , CNN{l}.stride , CNN{l}.pad_size );
            end
        end
    elseif strcmp( CNN{l}.type, 'full_connection' )
        if CNN{l-1}.vector == true
            CNN{l}.X = bsxfun( @plus, CNN{l-1}.X * CNN{l}.W, CNN{l}.bias );
        else
            map_size = prod( CNN{l-1}.map_size );
            CNN{l-1}.X_vec = zeros( N, map_size * CNN{l-1}.output );
            for i = 1 : CNN{l-1}.output
                CNN{l-1}.X_vec( : , (i-1) * map_size + 1 : i * map_size ) = reshape( CNN{l-1}.X{i}, map_size, N )';
            end
            CNN{l}.X = bsxfun( @plus, CNN{l-1}.X_vec * CNN{l}.W, CNN{l}.bias );
        end
    elseif strcmp( CNN{l}.type, 'activation' )
        f = str2func( CNN{l}.activation );
        if CNN{l-1}.vector == true
            CNN{l}.X = f( CNN{l-1}.X );
        else
            for j = 1 : CNN{l}.output
                CNN{l}.X{j} = f( CNN{l-1}.X{j} );
            end
        end
    else
        error('layer type wrong')
    end
end

end