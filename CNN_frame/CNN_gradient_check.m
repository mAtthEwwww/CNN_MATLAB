function CNN_gradient_check( input , target , CNN , cost_function , check_size , epsilon , tolerance )

cost_function = str2func(cost_function);

[~, ~, N] = size(input{1});

disorder = randperm( N );

[ input, target ] = get_mini_batch( input, target, disorder( 1 : check_size ) );

CNN = CNN_feedforward(CNN, input, true);

CNN = CNN_backpropagation(CNN, target);

L = length( CNN );

for l = L : -1 : 2
    fprintf('checking layer: %d, type: %s\n', l, CNN{l}.type)
    if strcmp( CNN{l}.type, 'convolution' )
        for j = 1 : CNN{l}.output
            for i = 1 : CNN{l-1}.output
                for r = 1 : CNN{l}.weight.shape(1)
                    for c = 1 : CNN{l}.weight.shape(2)
                        check = CNN;
                        check{l}.weight.kernel{j,i}( r, c ) = CNN{l}.weight.kernel{j,i}( r, c ) - epsilon;
                        check = CNN_feedforward(check, input, true);
                        left = cost_function(check{L}.X, target);
                        check{l}.weight.kernel{j,i}( r, c ) = CNN{l}.weight.kernel{j,i}( r, c ) + epsilon;
                        check = CNN_feedforward(check, input, true);
                        right = cost_function( check{L}.X, target );
                        numerical_grad_kernel = ( right - left ) / ( 2 * epsilon );
                        if abs(numerical_grad_kernel - CNN{l}.weight.grad{j,i}( r, c ) ) > tolerance
                            fprintf('gradient check fail at convolution layer: %d, kernel: (%d, %d), row: %d, column: %d\n', l, j, i, r, c) 
                            fprintf('numerical:  %.18f\n', numerical_grad_kernel)
                            fprintf('calculated: %.18f\n', (CNN{l}.weight.grad{j,i}( r , c )))
                        end
                    end
                end
            end
        end

        if CNN{l}.bias.option == true
            for j = 1 : CNN{l}.output
                check = CNN;
                check{l}.bias.b{j} = CNN{l}.bias.b{j} - epsilon;
                check = CNN_feedforward(check, input, true);
                left = cost_function( check{L}.X, target );
                check{l}.bias.b{j} = CNN{l}.bias.b{j} + epsilon;
                check = CNN_feedforward(check, input, true);
                right = cost_function( check{L}.X, target );
                numerical_grad_bias = ( right - left ) / ( 2 * epsilon );
                if abs( numerical_grad_bias - CNN{l}.bias.grad{j} ) > tolerance
                    fprintf('gradient check fail at convolution layer: %d, bias: %d\n', l, j)
                    fprintf('numerical:  %.18f\n', numerical_grad_bias)
                    fprintf('calculated: %.18f\n', CNN{l}.bias.grad{j})
               end
            end
        end
    
    elseif strcmp(CNN{l}.type, 'batch_normalization')
        for j = 1 : CNN{l}.output
            check = CNN;
            check{l}.gamma.g{j} = CNN{l}.gamma.g{j} - epsilon;
            check = CNN_feedforward(check, input, true);
            left = cost_function(check{L}.X, target);
            check{l}.gamma.g{j} = CNN{l}.gamma.g{j} + epsilon;
            check = CNN_feedforward(check, input, true);
            right = cost_function(check{L}.X, target);
            numerical_grad = (right - left) / (2 * epsilon);
            if abs(numerical_grad - CNN{l}.gamma.grad{j}) > tolerance
                fprintf('gradient check fail at batch normalization layer: %d, gamma: %d\n', l, j)
                fprintf('numerical:  %.18f\n', numerical_grad)
                fprintf('calculated: %.18f\n', CNN{l}.gamma.grad{j})
            end
        end
        
        for j = 1 : CNN{l}.output
            check = CNN;
            check{l}.beta.b{j} = CNN{l}.beta.b{j} - epsilon;
            check = CNN_feedforward(check, input, true);
            left = cost_function(check{L}.X, target);
            check{l}.beta.b{j} = CNN{l}.beta.b{j} + epsilon;
            check = CNN_feedforward(check, input, true);
            right = cost_function(check{L}.X, target);
            numerical_grad = (right - left) / (2 * epsilon);
            if abs(numerical_grad - CNN{l}.beta.grad{j}) > tolerance
                fprintf('gradient check fail at batch normalization layer: %d, beta: %d\n', l, j)
                fprintf('numerical:  %.18f\n', numerical_grad)
                fprintf('calculated: %.18f\n', CNN{l}.beta.grad{j})
            end
        end
        
    elseif strcmp( CNN{l}.type, 'full_connection' )
        if CNN{l-1}.isTensor == true
            fan_in = CNN{l-1}.output * prod( CNN{l-1}.map_size );
        else
            fan_in = CNN{l-1}.output;
        end

        for r = 1 : CNN{l}.output
            for c = 1 : fan_in
                check = CNN;
                check{l}.weight.W( r, c ) = CNN{l}.weight.W( r, c ) - epsilon;
                check = CNN_feedforward(check, input, true);
                left = cost_function( check{L}.X, target );
                check{l}.weight.W( r, c ) = CNN{l}.weight.W( r, c ) + epsilon;
                check = CNN_feedforward(check, input, true);
                right = cost_function( check{L}.X, target );
                numerical_grad_W = ( right - left ) / ( 2 * epsilon );
                if abs( numerical_grad_W - CNN{l}.weight.grad( r, c ) ) > tolerance
                    fprintf('gradient check fail at full connection layer: %d, weight: (%d, %d)\n', l, r, c)
                    fprintf('numerical:  %.18f\n', numerical_grad_W)
                    fprintf('calculated: %.18f\n', CNN{l}.weight.grad( r, c ))
                end
            end
        end
        for c = 1 : CNN{l}.output
            check = CNN;
            check{l}.bias.b(1, c) = CNN{l}.bias.b(1, c) - epsilon;
            check = CNN_feedforward(check, input, true);
            left = cost_function( check{L}.X, target );
            check{l}.bias.b(1, c) = CNN{l}.bias.b(1, c) + epsilon;
            check = CNN_feedforward(check, input, true);
            right = cost_function( check{L}.X, target );
            numerical_grad_bias = ( right - left ) / ( 2 * epsilon );
            if abs( numerical_grad_bias - CNN{l}.bias.grad(1, c) ) > tolerance
                fprintf('gradient check fail at full connection layer: %d, bias: %d\n', l, c)
                fprintf('numerical:  %.18f\n', numerical_grad_bias)
                fprintf('calculated: %.18f\n', CNN{l}.bias.grad(1, c))
            end
        end
    end
end

end
