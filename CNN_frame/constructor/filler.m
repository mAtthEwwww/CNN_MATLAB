function weight = filler( input , output , size , weight_filler )

if strcmp( weight_filler.type , 'xavier' )
    fan_in = input * prod( size );
    fan_out = output * prod( size );
    weight = sqrt( 6 / ( fan_in + fan_out ) ) * ( 2 * rand( size ) - 1 );
elseif strcmp( weight_filler.type , 'gaussian' )
    weight = weight_filler.std * randn( size );
elseif strcmp( weight_filler.type , 'msra' )
    fan_in = input * prod( size );
    weight = sqrt( 2 / fan_in ) * randn( size );
elseif strcmp( weight_filler.type , 'msra1' )
    fan_in = input * prod( size );
    weight = sqrt( 1 / fan_in );
else
    error('weight filler wrong');
end

end
