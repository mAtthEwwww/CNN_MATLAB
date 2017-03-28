function CNN = CNN_initialization( CNN )

for l = 2 : 1 : length( CNN );
    if strcmp( CNN{l}.type, 'convolution' )
        CNN{l} = convolution_constructor( CNN{l}, CNN{l-1}.output, CNN{l-1}.map_size );
    elseif strcmp( CNN{l}.type, 'sampling' )
        CNN{l} = sampling_constructor( CNN{l}, CNN{l-1}.output, CNN{l-1}.map_size );
    elseif strcmp( CNN{l}.type, 'full_connection' )
        CNN{l} = full_connection_constructor( CNN{l}, CNN{l-1} );
    elseif strcmp( CNN{l}.type, 'activation' )
        CNN{l} = activation_constructor( CNN{l}, CNN{l-1} );
    else
        error('layer type wrong')
    end
end

end