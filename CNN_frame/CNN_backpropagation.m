function CNN = CNN_backpropagation( target , CNN )
% CNN_backpropagation.m
% backpropagation with target
%
% Inputs:
%       target  is a bool matrix, each row is a one-hot label of a sample 
%       CNN  is a array of cell, each cell is a layer of CNN
%
% Outputs:
%       CNN  is a array of cell


% extract the sample size
[ ~, ~, N ] = size(CNN{1}.X{1});

% extract the number of layer
L = length( CNN );

% calculate the delta of the final layer
CNN{L}.delta = CNN{L}.X - target;

% calculate the delta from back to front, layer by layer
for l = L : -1 : 2
    % if layer l is convolution layer, do the transpose convolution (or deconvolution)
    if strcmp( CNN{l}.type, 'convolution' )
        % if expand is true
        if CNN{l}.expand == true
            for j = 1 : CNN{l}.output
                % flip the tensor delta
                tmp_delta = CNN{l}.delta{j}(:, :, end:-1:1);
                % calculate the gradient of weight, or convolution kernel
                for i = 1 : CNN{l-1}.output
                    CNN{l}.grad_kernel{i}{j} = convn_pad( rot90( CNN{l-1}.X{i}, 2 ), tmp_delta , CNN{l}.pad ) / N;
                end
                % calculate the the gradient of bias
                CNN{l}.grad_bias{j} = sum( CNN{l}.delta{j}( : ) ) / N;
            end
            % if layel is not the second layer, calculate the delta in layer l-1
            if l ~= 2
                for i = 1 : CNN{l-1}.output
                    CNN{l-1}.delta{i} = zeros( [ CNN{l-1}.map_size , N ] );
                    for j = 1 : CNN{l}.output
                        CNN{l-1}.delta{i} = CNN{l-1}.delta{i} + convn_pad( CNN{l}.delta{j}, rot90( CNN{l}.kernel{i}{j}, 2 ) , CNN{l}.pad );
                    end
                end
            end
        % else if expand is false 
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
    % if layer l is sampling layer, do the up sampling
    elseif strcmp( CNN{l}.type, 'sampling' )
        % if sampling method is max
        if strcmp( CNN{l}.method, 'max' )
            for j = 1 : CNN{l}.output
                CNN{l-1}.delta{j} = up_max_sampling( CNN{l}.delta{j}, CNN{l}.max_position{j}, CNN{l-1}.map_size, CNN{l}.sampling_size, CNN{l}.stride , CNN{l}.pad_size );
            end
        % if sampling method is average
        elseif strcmp( CNN{l}.method, 'average' )
            for j = 1 : CNN{l}.output
                CNN{l-1}.delta{j} = up_average_sampling( CNN{l}.delta{j} , CNN{l-1}.map_size , CNN{l}.sampling_size , CNN{l}.stride , CNN{l}.pad_size );
            end
        else
            error('sampling method wrong')
        end
    % if layer l is full connection layer
    elseif strcmp( CNN{l}.type, 'full_connection' )
        % calculate the gradient of weight
        if CNN{l-1}.vector == true
            CNN{l}.grad_W = CNN{l-1}.X' * CNN{l}.delta / N;
        else
            CNN{l}.grad_W = CNN{l-1}.X_vec' * CNN{l}.delta / N;
        end
        % calculate the gradient of bias
        CNN{l}.grad_bias = sum( CNN{l}.delta, 1 ) / N;
        % if layer l is not the second layer, calculate the delta in layer l-1
        if l ~= 2
            % calculate the delta
            delta = CNN{l}.delta * CNN{l}.W';
            % if the shape of layer l-1 is vector (or matrix with each row is a feature vector of a sample)
            if CNN{l-1}.vector == true
                % then copy the delta directly
                CNN{l-1}.delta = delta;
            % else if the shape is matrix (or tensor, height x width x sample size)
            else
                % reshape the delta
                map_size = prod( CNN{l-1}.map_size );
                for i = 1 : CNN{l-1}.output
                    CNN{l-1}.delta{i} = reshape( delta( : , (i-1) * map_size + 1 : i * map_size )', [ CNN{l-1}.map_size , N ] );
                end
            end
        end

    % if layer l is activation layer, then multiply delta by derived activation
    elseif strcmp( CNN{l}.type, 'activation' )
        % transfer the string of activation to function handle of derivatives
        derived_f = str2func([ 'derived_' , CNN{l}.activation ]);
        if CNN{l-1}.vector == true
            CNN{l-1}.delta = derived_f( CNN{l-1}.X ) .* CNN{l}.delta;
        else
            for i = 1 : CNN{l-1}.output
                CNN{l-1}.delta{i} = derived_f( CNN{l-1}.X{i} ) .* CNN{l}.delta{i};
            end
        end
    end
end
        
end
