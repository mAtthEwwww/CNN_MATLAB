function [ data , labels ] = prepare( number , std )

number_1 = ceil( 0.5 * number );

number_2 = number - number_1;

data_1 = generate_data( number_1 , 1 , std );

label_1 = [ ones( number_1 , 1 ) , zeros( number_1 , 1 ) ];

data_2 = generate_data( number_2 , 0 , std );

label_2 = [ zeros( number_2 , 1 ) , ones( number_2 , 1 ) ];

data = [data_1;data_2];

labels = [label_1;label_2];

disorder = randperm( length(labels) );

data = data(disorder,:);

labels = labels( disorder , : );

end