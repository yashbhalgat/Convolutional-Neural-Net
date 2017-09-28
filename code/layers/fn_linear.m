% ----------------------------------------------------------------------
% input: num_in x batch_size
% output: num_out x batch_size
% hyper_params:
% params.W: num_out x num_in
% params.b: num_out x 1
% dv_output: same as output
% dv_input: same as input
% grad: same as params
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_linear(input, params, hyper_params, backprop, dv_output)

[num_in,batch_size] = size(input);
assert(num_in == hyper_params.num_in,...
	sprintf('Incorrect number of inputs provided at linear layer.\nGot %d inputs expected %d.',num_in,hyper_params.num_in));

output = zeros(hyper_params.num_out, batch_size);
% TODO: FORWARD CODE
bias = params.b*ones(1,batch_size);
x_hat = params.W*input;
output = x_hat + bias;

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    trans_W = (params.W).';
    dv_input = trans_W*dv_output;  % dv_input = dy/dx*output of previous layer
    trans_in = input.';
    grad.W = dv_output*trans_in;   % gradient for W = dy/dW*output of previous layer
	grad.b = sum(dv_output,2);  % gradient for b = dy/db*output of previous layer
end
