function layer = convolution_constructor( layer , input, input_map_size )

if layer.expand == true
    layer.map_size = input_map_size;
else
    layer.map_size = input_map_size - layer.kernel_size + 1;
end

layer.vector = false;

for j = 1 : layer.output
    for i = 1 : input
        layer.kernel{i}{j} = filler( input , layer.output , [ layer.kernel_size(1) , layer.kernel_size(2) ] , layer.weight_filler);
        layer.m_grad_kernel{i}{j} = zeros([layer.kernel_size(1), layer.kernel_size(2)] );
    end
    layer.bias{j} = 0;
    layer.m_grad_bias{j} = 0;
end

end
