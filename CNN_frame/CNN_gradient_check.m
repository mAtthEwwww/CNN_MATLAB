function CNN_gradient_check( input , target , CNN , cost_function , check_size , epsilon , tolerance )

cost_function = str2func( cost_function );

[ ~ , ~ , N ] = size( input{ 1 } );

disorder = randperm( N );

[ input, target ] = get_mini_batch( input, target, disorder( 1 : check_size ) );

CNN = CNN_feedforward( input , CNN );

CNN = CNN_backprogagation( target , CNN );

L = length( CNN );

for l = L : -1 : 2
    disp( l )
    if strcmp( CNN{l}.type, 'convolution' )
        for j = 1 : CNN{l}.output
            for i = 1 : CNN{l-1}.output
                for r = 1 : CNN{l}.kernel_size(1)
                    for c = 1 : CNN{l}.kernel_size(2)
                        check = CNN;
                        check{l}.kernel{i}{j}( r, c ) = CNN{l}.kernel{i}{j}( r, c ) - epsilon;
                        check = CNN_feedforward( input, check );
                        left = cost_function( check{L}.X, target );
                        check{l}.kernel{i}{j}( r, c ) = CNN{l}.kernel{i}{j}( r, c ) + epsilon;
                        check = CNN_feedforward( input, check );
                        right = cost_function( check{L}.X, target );
                        numecal_grad_kernel = ( right - left ) / ( 2 * epsilon );
                        if abs( numecal_grad_kernel - CNN{l}.grad_kernel{i}{j}( r, c ) ) > tolerance
                            disp(numecal_grad_kernel)
                            disp(CNN{l}.grad_kernel{i}{j}( r , c ))
                            disp(['convolution kernel fail: ' , num2str(l) , '²ã ' , num2str(j) , 'map ' , num2str(i) , 'inputmap ' , num2str(r) , 'ĞĞ ' , num2str(c) , 'ÁĞ' ])
                        end
                    end
                end
            end
            check = CNN;
            check{l}.bias{j} = CNN{l}.bias{j} - epsilon;
            check = CNN_feedforward( input, check );
            left = cost_function( check{L}.X, target );
            check{l}.bias{j} = CNN{l}.bias{j} + epsilon;
            check = CNN_feedforward( input, check );
            right = cost_function( check{L}.X, target );
            numecal_grad_bias = ( right - left ) / ( 2 * epsilon );
            if abs( numecal_grad_bias - CNN{l}.grad_bias{j} ) > tolerance
                disp(numecal_grad_bias)
                disp(CNN{l}.grad_bias{j})
                disp(['convolution bias fail: ' , num2str(l) , '²ã ' , num2str(j) , 'map '])
            end
        end
        
    elseif strcmp( CNN{l}.type, 'full_connection' )
        if CNN{l-1}.vector == true
            R = CNN{l-1}.output;
        else
            R = CNN{l-1}.output * prod( CNN{l-1}.map_size );
        end
        for r = 1 : R
            for c = 1 : CNN{l}.output
                check = CNN;
                check{l}.W( r, c ) = CNN{l}.W( r, c ) - epsilon;
                check = CNN_feedforward( input, check );
                left = cost_function( check{L}.X, target );
                check{l}.W( r, c ) = CNN{l}.W( r, c ) + epsilon;
                check = CNN_feedforward( input, check );
                right = cost_function( check{L}.X, target );
                numecal_grad_W = ( right - left ) / ( 2 * epsilon );
                if abs( numecal_grad_W - CNN{l}.grad_W( r, c ) ) > tolerance
                    disp(numecal_grad_W)
                    disp(CNN{l}.grad_W( r, c ))
                    disp(['full weight fail: ' , num2str(l) , '²ã ' , num2str(r), 'c' , num2str(c) , 'c' ])
                end
            end
        end
        for c = 1 : CNN{l}.output
            check = CNN;
            check{l}.bias( 1, c ) = CNN{l}.bias( 1, c ) - epsilon;
            check = CNN_feedforward( input, check );
            left = cost_function( check{L}.X, target );
            check{l}.bias( 1, c ) = CNN{l}.bias( 1, c ) + epsilon;
            check = CNN_feedforward( input, check );
            right = cost_function( check{L}.X, target );
            numecal_grad_bias = ( right - left ) / ( 2 * epsilon );
            if abs( numecal_grad_bias - CNN{l}.grad_bias( 1, c ) ) > tolerance
                disp(numecal_grad_bias)
                disp(CNN{l}.grad_bias( 1, c ))
                disp(['full bias fail: ' , num2str(l) , '²ã ', num2str(c) , 'c' ])
            end
        end
    end
end

end
