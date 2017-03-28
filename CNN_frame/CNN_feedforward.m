function CNN = CNN_feedforward( X_input , CNN )
% CNN_feedforward.m
% the feedforward function of CNN
% Inputs:
%       X_input  is an array of cell, each cell contains a 3-order tensor
%               (height x width x sample-size) and represent a color channel 
%       CNN  is an array of cell, each cell is a layer of CNN
% Outputs:
%       CNN  is an array of cell, the feedforward result is stored in the last layer (CNN{length(CNN)}.X)

% extract the sample size (the number of sample)
[ ~ , ~ , N ] = size(X_input{1});

% store the input set in the input layer
for j = 1 : CNN{1}.output
    CNN{1}.X{j} = X_input{j};
end

% do the feedforward layer by layer
for l = 2 : length(CNN)
    % if layer l is convolution layer, then do the convolution
    if strcmp( CNN{l}.type, 'convolution' )
        % if expand is true, then do the convolution with zero padding
        if CNN{l}.expand == true
            % calculate the convolution result channel by channel
            for j = 1 : CNN{l}.output
                % construct an empty tensor (height x width x sample size)
                % and add the bias
                CNN{l}.X{j} = zeros( [ CNN{l}.map_size , N ] ) + CNN{l}.bias{j};
                for i = 1 : CNN{l-1}.output
                    CNN{l}.X{j} = CNN{l}.X{j} + convn_pad( CNN{l-1}.X{i}, CNN{l}.kernel{i}{j} , CNN{l}.pad );
                end
            end
        % else do the convolution wiout zero padding
        else
            for j = 1 : CNN{l}.output
                CNN{l}.X{j} = zeros( [ CNN{l}.map_size , N ] ) + CNN{l}.bias{j};
                for i = 1 : CNN{l-1}.output
                    CNN{l}.X{j} = CNN{l}.X{j} + convn( CNN{l-1}.X{i}, CNN{l}.kernel{i}{j}, 'valid' );
                end
            end
        end

    % if layer l is sampling layer, then do the down sampling
    elseif strcmp( CNN{l}.type, 'sampling' )
        % do the down sampling channel by channel
        for j = 1 : CNN{l}.output
            % if sampling method is max, do the max sampling
            if strcmp( CNN{l}.method, 'max' )
                [ CNN{l}.X{j} , CNN{l}.max_position{j} ] = down_max_sampling( CNN{l-1}.X{j}, CNN{l}.map_size, CNN{l}.sampling_size, CNN{l}.stride, CNN{l}.pad_size );
            % else if sampling method is average, do the average sampling
            elseif strcmp( CNN{l}.method, 'average' )
                CNN{l}.X{j} = down_average_sampling( CNN{l-1}.X{j} , CNN{l}.map_size , CNN{l}.sampling_size , CNN{l}.stride , CNN{l}.pad_size );
            end
        end

    % if layer l is full connection layer, then do the inner product
    elseif strcmp( CNN{l}.type, 'full_connection' )
        if CNN{l-1}.vector == true
            % if the shape of l-1 layer is vector (or matrix, each row is a feature vector of a sample), then directly do the inner product and add bias
            CNN{l}.X = bsxfun( @plus, CNN{l-1}.X * CNN{l}.W, CNN{l}.bias );
            % else reshape the image matrix (or tensor) to a feature vector (or matrix, each row is a feature vector)
        else
            % calculate the dimension of feature vector in layer l-1
            map_size = prod( CNN{l-1}.map_size );
            % construct an empty matrix for the feature vector, each row is a sample
            CNN{l-1}.X_vec = zeros( N, map_size * CNN{l-1}.output );
            % reshape the image matrix channel by channel
            for i = 1 : CNN{l-1}.output
                CNN{l-1}.X_vec( : , (i-1) * map_size + 1 : i * map_size ) = reshape( CNN{l-1}.X{i}, map_size, N )';
            end
            % do the inner product and add the bias
            CNN{l}.X = bsxfun( @plus, CNN{l-1}.X_vec * CNN{l}.W, CNN{l}.bias );
        end

    % if layer l is activation layer, then do the activation
    elseif strcmp( CNN{l}.type, 'activation' )
        % convert the string of activation function to function handle
        f = str2func( CNN{l}.activation );
        % if the shape of layer l-1 is vector, do the activation directly
        if CNN{l-1}.vector == true
            CNN{l}.X = f( CNN{l-1}.X );
        % else do the activation channel by channel
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
