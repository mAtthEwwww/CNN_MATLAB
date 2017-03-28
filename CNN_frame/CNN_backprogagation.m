function CNN = CNN_backprogagation( target , CNN )

[ ~, ~, N ] = size( CNN{1}.X{1} );
L = length( CNN );

CNN{L}.delta = CNN{L}.X - target;

for l = L : -1 : 2
    if strcmp( CNN{l}.type, 'convolution' )
        if CNN{l}.expand == true
            for j = 1 : CNN{l}.output
                tmp_delta = CNN{l}.delta{j}(:,:,end:-1:1);
                for i = 1 : CNN{l-1}.output
                    CNN{l}.grad_kernel{i}{j} = convn_pad( rot90( CNN{l-1}.X{i}, 2 ), tmp_delta , CNN{l}.pad ) / N;
                end
                CNN{l}.grad_bias{j} = sum( CNN{l}.delta{j}( : ) ) / N;
            end
            if l ~= 2
                for i = 1 : CNN{l-1}.output
                    CNN{l-1}.delta{i} = zeros( [ CNN{l-1}.map_size , N ] );
                    for j = 1 : CNN{l}.output
                        CNN{l-1}.delta{i} = CNN{l-1}.delta{i} + convn_pad( CNN{l}.delta{j}, rot90( CNN{l}.kernel{i}{j}, 2 ) , CNN{l}.pad );
                    end
                end
            end
        else
            for j = 1 : CNN{l}.output
                tmp_delta = CNN{l}.delta{j}(:,:,end:-1:1);
                for i = 1 : CNN{l-1}.output
                    CNN{l}.grad_kernel{i}{j} = convn( rot90( CNN{l-1}.X{i}, 2 ), tmp_delta , 'valid' ) / N;
                end
                CNN{l}.grad_bias{j} = sum( CNN{l}.delta{j}( : ) ) / N;
            end
            if l ~= 2
                for i = 1 : CNN{l-1}.output
                    CNN{l-1}.delta{i} = zeros( [ CNN{l-1}.map_size , N ] );
                    for j = 1 : CNN{l}.output
                        CNN{l-1}.delta{i} = CNN{l-1}.delta{i} + convn( CNN{l}.delta{j}, rot90( CNN{l}.kernel{i}{j}, 2 ) , 'full' );
                    end
                end
            end
        end
    elseif strcmp( CNN{l}.type, 'sampling' )
        if strcmp( CNN{l}.method, 'max' )
            for j = 1 : CNN{l}.output
                CNN{l-1}.delta{j} = up_max_sampling( CNN{l}.delta{j}, CNN{l}.max_position{j}, CNN{l-1}.map_size, CNN{l}.sampling_size, CNN{l}.stride , CNN{l}.pad_size );
            end
        elseif strcmp( CNN{l}.method, 'average' )
            for j = 1 : CNN{l}.output
%                 size(CNN{l}.sampling_size)
                CNN{l-1}.delta{j} = up_average_sampling( CNN{l}.delta{j} , CNN{l-1}.map_size , CNN{l}.sampling_size , CNN{l}.stride , CNN{l}.pad_size );
            end
        else
            error('sampling method wrong')
        end
    elseif strcmp( CNN{l}.type, 'full_connection' )
        if CNN{l-1}.vector == true
            CNN{l}.grad_W = CNN{l-1}.X' * CNN{l}.delta / N;
        else
            CNN{l}.grad_W = CNN{l-1}.X_vec' * CNN{l}.delta / N;
        end
        CNN{l}.grad_bias = sum( CNN{l}.delta, 1 ) / N;
        if l ~= 2
            delta = CNN{l}.delta * CNN{l}.W';
            if CNN{l-1}.vector == true
                CNN{l-1}.delta = delta;
            else
                map_size = prod( CNN{l-1}.map_size );
                for i = 1 : CNN{l-1}.output
                    CNN{l-1}.delta{i} = reshape( delta( : , (i-1) * map_size + 1 : i * map_size )', [ CNN{l-1}.map_size , N ] );
                end
            end
        end
    elseif strcmp( CNN{l}.type, 'activation' )
        diff_f = str2func([ 'diff_' , CNN{l}.activation ]);
        if CNN{l-1}.vector == true
            CNN{l-1}.delta = diff_f( CNN{l-1}.X ) .* CNN{l}.delta;
        else
            for i = 1 : CNN{l-1}.output
                CNN{l-1}.delta{i} = diff_f( CNN{l-1}.X{i} ) .* CNN{l}.delta{i};
            end
        end
    end
end
        
end