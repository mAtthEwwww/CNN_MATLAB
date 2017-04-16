function layer = activation_constructor( layer, input_layer )

if input_layer.isTensor == true
    layer.isTensor = true;
    layer.map_size = input_layer.map_size;
    layer.X = cell(input_layer.output, 1);
else
    layer.isTensor= false;
    layer.X = [];
end

layer.output = input_layer.output;

end
