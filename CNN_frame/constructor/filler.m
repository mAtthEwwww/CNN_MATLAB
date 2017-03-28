function weight = filler( input , output , size , weight_filler , std )

if strcmp( weight_filler , 'xavier' )
    fan_in = input * prod( size );
    fan_out = output * prod( size );
    weight = sqrt( 6 / ( fan_in + fan_out ) ) * ( 2 * rand( size ) - 1 );
elseif strcmp( weight_filler , 'gaussian' )
    weight = std * randn( size );
elseif strcmp( weight_filler , 'msra' ) || strcmp( weight_filler , 'msra2' )
    fan_in = input * prod( size );
    weight = sqrt( 2 / fan_in ) * randn( size );
elseif strcmp( weight_filler , 'msra1' )
    fan_in = input * prod( size );
    weight = sqrt( 1 / fan_in );
else
    error('weight filler wrong');
end

end