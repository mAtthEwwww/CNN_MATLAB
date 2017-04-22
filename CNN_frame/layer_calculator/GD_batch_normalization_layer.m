function layer = GD_batch_normalization_layer( layer, lr, m_rate )

for j = 1 : length(layer.X)
    
    layer.gamma.momentum{j} = m_rate * layer.gamma.momentum{j} + (1 - m_rate) * lr * layer.gamma.grad{j};

    layer.gamma.g{j} = layer.gamma.g{j} - layer.gamma.momentum{j};

    layer.beta.momentum{j} = m_rate * layer.beta.momentum{j} + (1 - m_rate) * lr * layer.beta.grad{j};

    layer.beta.b{j} = layer.beta.b{j} - layer.beta.momentum{j};

end

end
