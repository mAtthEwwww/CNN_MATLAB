function layer = activation_constructor( layer, input_layer )

if input_layer.vector == true
    layer.vector = true;
else
    layer.vector = false;
    layer.map_size = input_layer.map_size;
end

layer.output = input_layer.output;

end