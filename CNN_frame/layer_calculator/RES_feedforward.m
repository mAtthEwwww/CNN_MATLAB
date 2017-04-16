function layer = RES_feedforward( layer , X_1 , X_2 )

for j = 1 : length(X_1)
    
    layer.X{j} = X_1{j};

end

for i = 1 : length(X_2)

    layer.X{i} = layer.X{i} + X_2{i};

end

end
