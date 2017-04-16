function CNN = CNN_initialization( CNN )
% CNN_initialization.m
% this function will completing the structure of CNN
%
% Inputs:
%       CNN  is an array of cell, each cell is a layer of CNN
% Outputs:
%       CNN  is an array of cell

for l = 2 : 1 : length( CNN );
    % if layer l is a convolution layer, then call the convolution constructor
    if strcmp(CNN{l}.type, 'convolution')
        CNN{l} = convolution_constructor(CNN{l}, CNN{l-1}.output, CNN{l-1}.map_size);

    % if layer l is a residual block, then call the residual constructor
    elseif strcmp(CNN{l}.type, 'residual_block')
        CNN{l} = residual_block_constructor(CNN{l}, CNN{l-1}.output, CNN{l-1}.map_size);

    % if layer l is a batch normalization layer, then call the batch normalization constructor
    elseif strcmp(CNN{l}.type, 'batch_normalization')
        CNN{l} = batch_normalization_constructor(CNN{l}, CNN{l-1}.output, CNN{l-1}.map_size);

    % if layer l is a sampling layer, then call the sampling constructor
    elseif strcmp( CNN{l}.type, 'sampling' )
        CNN{l} = sampling_constructor(CNN{l}, CNN{l-1}.output, CNN{l-1}.map_size);

    % if layer l is a full connection layer, then call the full connection constructor
    elseif strcmp( CNN{l}.type, 'full_connection' )
        CNN{l} = full_connection_constructor( CNN{l}, CNN{l-1} );

    % if layer l is a activation layer, then call the activation constructor
    elseif strcmp( CNN{l}.type, 'activation' )
        CNN{l} = activation_constructor( CNN{l}, CNN{l-1} );
        
    else
    % otherwise
        error('layer type wrong')
    end
end

end
