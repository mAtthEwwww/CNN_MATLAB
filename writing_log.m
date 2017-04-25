function writing_log( CNN , result, config )

struct_str = sprintf('layer 1   type: input   width:%i\r\n' , CNN{ 1 }.output);
for l = 2 : length(CNN)
    if strcmp(CNN{l}.type , 'convolution')
        struct_str = sprintf( '%slayer %i   type: convolution    kernel size: %ix%i   bias option: %d   width: %i\r\n', struct_str, l, CNN{l}.weight.shape(1), CNN{l}.weight.shape(2), CNN{l}.bias.option, CNN{l}.output);
    elseif strcmp( CNN{l}.type , 'sampling' )
        struct_str = sprintf( '%slayer %i   type: sampling     stride: %ix%i   sampling size: %ix%i   sampling type: %s\r\n', struct_str, l, CNN{l}.sampling.stride(1), CNN{l}.sampling.stride(2), CNN{l}.sampling.shape(1), CNN{l}.sampling.shape(2), CNN{ l }.sampling.type);
    elseif strcmp(CNN{l}.type, 'batch_normalization')
        struct_str = sprintf('%slayer %i   type: batch normalization   moving average rate: %.3f\r\n', struct_str, l, CNN{l}.BN_decay);
    elseif strcmp(CNN{l}.type , 'full_connection')
        struct_str = sprintf('%slayer %i   type: full_connection   %dropout option: %d   width: %i\r\n', struct_str, l, CNN{l}.dropout.option, CNN{ l }.output);
    elseif strcmp(CNN{l}.type, 'activation')
        struct_str = sprintf( '%slayer %i   type: activation             %s\r\n' , struct_str, l, CNN{l}.activation );
    end
end
run_str = sprintf('Accuracy %.2f%%   cost %f   time %.1fs   epochs %i   learning rate %.7f   batchsize %i   momentum %.1f   half life %i', result.accuracy * 100, result.cost, result.train_time, result.epochs, config.learning_rate, config.batch_size, config.momentum, config.half_life);
log_str = sprintf( '%s\r\n%s' , run_str , struct_str );                 

% output the running log
fid = fopen('Log/Log', 'a');
fprintf(fid, '%s\r\n\r\n', log_str );
fclose(fid);

title_str = sprintf( 'AC:%.2f%%, cost:%.4f, TIME:%.1fs, EPOCHS:%i, LR:%.7f, BATCHSIZE:%i', ...
     result.accuracy*100, result.cost, result.train_time, result.epochs, config.learning_rate, config.batch_size);

title( title_str );
ylim([0 2.5]);

% output the figure
saveas( gcf , sprintf( '%s%s%s' , 'Log/' , run_str , '.png' ) );
