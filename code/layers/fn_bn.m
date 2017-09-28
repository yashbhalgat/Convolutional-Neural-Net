% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (filter_depth)
% params.W: filter_depth x 1
% params.b: filter_depth x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_bn(input, params, hyper_params, backprop, dv_output)
ep = 1e-5; % for stability
[out_height,out_width,num_channels,batch_size] = size(input);
assert(hyper_params.filter_depth == num_channels, 'Filter depth does not match number of input channels');
output = zeros(out_height,out_width,num_channels,batch_size);
% TODO: FORWARD CODE
means = zeros(num_channels);
var_channel = zeros(num_channels);
output_mid = output;
for indChannel = 1:num_channels
    means(indChannel) = mean(mean(mean(input(:,:,indChannel,:))));
    var_channel(indChannel) = var(var(var(input(:,:,indChannel,:))));
    output_mid(:,:,indChannel,:) = (input(:,:,indChannel,:)-means(indChannel))/sqrt(var_channel(indChannel)+ep);
    output(:,:,indChannel,:) = params.W(indChannel)*output_mid(:,:,indChannel,:)+params.b(indChannel);
end



dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
    dv_input = zeros(size(input));
    grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
    % TODO: BACKPROP CODE
    m = out_height*out_width*batch_size;
    Dy_Dxhat = zeros(dv_input);
    dv_var = zeros(num_channels);
    dy_means = zeros(num_channels);
    for indChannel = 1:num_channels
        d = dv_output(:,:,indChannel,:).*output_mid(:,:,indChannel,:);
        grad.W(indChannel) = sum(sum(sum(d(:,:,indChannel,:)));
        grad.b(indChannel) = sum(sum(sum(dv_output(:,:,indChannel,:))));
        Dy_Dxhat(:,:,indChannel,:) = dv_output(:,:,indChannel,:)*params.W(indChannel);
        dv_var(indChannel) = sum(reshape(Dy_Dxhat(:,:,indChannel,:).*(input(:,:,indChannel,:)-means(indChannel)),[],1))*-0.5 *(var_channel(indChannel)+ep)^(-1.5);
        dy_means(indChannel) = sum(reshape(Dy_Dxhat(:,:,indChannel,:)*(-1.0)/sqrt(var_channel(indChannel)+ep), [], 1))+dv_var(indChannel)*sum(reshape(-2*(input(:,:,indChannel,:)-means(indChannel)), [], 1))/m;
        dv_input(:,:,indChannel,:) = Dy_Dxhat(:,:,indChannel,:)/sqrt(var_channel(indChannel)+ep)+dv_var(indChannel)*2*(input(:,:,indChannel,:)-means(indChannel))/m+dy_means(indChannel)/m;
    end
end