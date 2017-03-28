function layer = full_connection_constructor( layer , layer_input )

layer.vector = true;

if layer_input.vector == true
    fan_in = layer_input.output;
else
    fan_in = layer_input.output * prod( layer_input.map_size );
end

layer.W = filler( fan_in, layer.output, [fan_in, layer.output], layer.weight_filler, layer.weight_std );
layer.bias = zeros( 1, layer.output );

layer.m_grad_W = zeros( fan_in, layer.output );
layer.m_grad_bias = zeros( 1 , layer.output );

end

