function hn = generate_data( number , true_or_false , noise_std )

f = 500;
tau = 1/500;
h=@(t,start_point)bsxfun(@ge,t,start_point).*exp(-(bsxfun(@minus,t,start_point)/tau)).*sin(2*pi*f.*bsxfun(@minus,t,start_point));

t = repmat(linspace(0.18, 0.22, 160),number,1);
start_point=0.03*(rand(number,1)+6);
ht=h(t,start_point);
n=normrnd(0,noise_std,[number , size(t,2)]);
if true_or_false
    hn =bsxfun(@plus, ht , n);
else
    hn = n;
end

% plot(t,hn,'r');
% hold on;
% plot(t,ht,'b')
% hold off

end