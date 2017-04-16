function [ mini_input , mini_target ] = get_mini_batch( input , target , index )

if nargin == 2
    index = target;
elseif nargin == 3
    mini_target = target( index , : );
end

mini_input = cell( 1 , length( input ) );
for j = 1 : length( input )
    mini_input{j} = input{j}( : , : , index );
end

end
