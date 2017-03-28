function CNN = CNN_train( train_input , train_target , validation_input , validation_target , config , CNN )

learning_rate = config.learning_rate;
half_life = config.half_life;
weight_decay = config.weight_decay;
max_epochs = config.max_epochs;
momentum = config.momentum;
batch_size = config.batch_size;
validate_interval = config.validate_interval;
cost_function = str2func( config.cost_function );
threshold = config.threshold;
[ ~ , ~ , N ] = size( train_input{ 1 } );

iterations = floor( N / batch_size );               %����ÿ��epoch����Ҫ�����Ĵ���
cost = zeros( 1 , max_epochs * iterations );        %����һ�����飬���ÿ�ε�������ʧ����ֵ
average_cost = zeros( 1 , max_epochs * iterations );%����һ����ֵ��������һ��epoch�ڵ�ƽ����ʧ����ֵ
validate_accuracy = zeros( 2 , floor( max_epochs * iterations / validate_interval )+1 ); %����һ�����飬���ÿ����֤���Ĳ�����ȷ��
iter = 0;                                           %���ü���������¼�ܵĵ�������
epochs = 0;                                         %���ü���������¼����ѵ�����ϵ�ѵ������
validated = 1;                                      %���ü���������¼����֤���Ľ��в��ԵĴ���
go_on = 1;                                          %����һ������ֹͣ�ı�ǣ�������Ϊ0����ֹͣ����
k = -log(2) / (iterations * half_life );            %���ݰ�˥�ڼ���ָ��˥���Ĳ���
while epochs < max_epochs && go_on
    epochs = epochs + 1;
    
    disorder = randperm( N );                       %ÿ�ζ�����ѵ��������ѵ��ʱ���ȴ���˳�򣬲���һ�������������
    sub_iter = 0;                                   %���ü���������¼�����epoch�ڵĵ�������
    while sub_iter < iterations && go_on
        
        iter = iter + 1;
        sub_iter = sub_iter + 1;
        lr = learning_rate * exp( k * iter );   %��ô��ѧϰ���ʽ���ָ��˥��
        
        if sub_iter < iterations                    %�����޷Żصķ�ʽ����ѵ�����г�ȡһ��mini-batch
            [batch_input, batch_target] = get_mini_batch( train_input, train_target, disorder( (sub_iter-1)*batch_size+1 : sub_iter*batch_size ) );
        else                                        %�����epoch�ڣ��������һ�ε�����ʱ�򣬽�ʣ������ѵ�����ݽ���ѵ��������batch_size��������ѵ�����������
            [batch_input, batch_target] = get_mini_batch( train_input, train_target, disorder( (sub_iter-1)*batch_size+1 : end ) );
        end
        
        CNN = CNN_feedforward( batch_input , CNN );                             %ǰ�򴫲�
        CNN = CNN_backprogagation( batch_target , CNN );                    %���򴫲�
        CNN = CNN_gradient_descent( CNN , momentum , lr , weight_decay );   %�����ݶ�
        
        if mod( iter, validate_interval ) == 0
            validated = validated + 1;                                      %ÿ����һ��������������֤�����в���
            [ accuracy , ~ ] = CNN_test( validation_input, validation_target, CNN );
            validate_accuracy( 1, validated ) = iter;
            validate_accuracy( 2, validated ) = accuracy;
        end
        cost( iter ) = cost_function( CNN{ length( CNN ) }.X , batch_target );        
        average_cost( iter ) = mean( cost( max( 1 , iter - iterations ) : iter ) );
        fprintf('Accuracy: %.4f, LR: %.7f, EPO: %i/%i, ITR: %i/%i, AVG: %f, CUR: %f\n' , validate_accuracy( 2, validated ) , lr , epochs , max_epochs , sub_iter , iterations , average_cost( iter ) , cost( iter ) );
        if average_cost( iter ) < threshold
            go_on = 0;
        end
    end
end

figure

[ img, ~, ~ ] = plotyy( 1 : iter, average_cost( 1 : iter ) , validate_accuracy( 1 , 1 : validated ) , validate_accuracy( 2 , 1 : validated ) , @plot );
set( get( img(1), 'ylabel' ), 'string', 'cost value', 'fontsize', 16 );
set( get( img(2), 'ylabel' ), 'string', 'AC', 'fontsize', 16 );
xlabel( 'iterations', 'fontsize', 16 );

CNN{1}.cost = average_cost( iter );
CNN{1}.epochs = epochs;

end
