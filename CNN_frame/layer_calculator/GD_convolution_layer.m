function layer = GD_convolution_layer( layer, lr , m_rate , w_decay )

w_lr = lr * layer.weight.learning_rate;

for j = 1 : size(layer.weight.kernel, 1)

    for i = 1 : size(layer.weight.kernel, 2)

        layer.weight.momentum{j, i} = m_rate * layer.weight.momentum{j, i} + (1 - m_rate) * w_lr * (layer.weight.grad{j, i} + w_decay * layer.weight.kernel{j, i});

        layer.weight.kernel{j, i} = layer.weight.kernel{j, i} - layer.weight.momentum{j, i};

    end

end

if layer.bias.option == true

    b_lr = lr * layer.bias.learning_rate;

    for j = 1 : size(layer.weight.kernel, 1)

        layer.bias.momentum{j} = m_rate * layer.bias.grad{j} + (1 - m_rate) * b_lr * layer.bias.grad{j};

        layer.bias.b{j} = layer.bias.b{j} - layer.bias.momentum{j};

    end
end

end
        
