function y = derived_sigmoid( a )

y = 1 ./ ( 1 + exp(-a) );

y = y .* ( 1 - y );

end
