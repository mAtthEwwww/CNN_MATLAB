function layer = GD_full_connection_layer( layer , lr , m_rate , w_decay )

w_lr = lr * layer.weight.learning_rate;

b_lr = lr * layer.bias.learning_rate;

layer.weight.momentum = m_rate * layer.weight.momentum + (1 - m_rate) * w_lr * (layer.weight.grad + w_decay * layer.weight.W);

layer.weight.W = layer.weight.W - layer.weight.momentum;

layer.bias.momentum = m_rate * layer.bias.momentum + (1 - m_rate) * b_lr * layer.bias.grad;

layer.bias.b = layer.bias.b - layer.bias.momentum;

end
