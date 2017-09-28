% ----------------------------------------------------------------------
% input: any n-d array
% output: same as input
% dv_output: same as input
% dv_input: same as input
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_relu(input, params, hyper_params, backprop, dv_output)
% Rectified linear unit activation function

% TODO: FORWARD CODE
output = max(input,0.01*input);

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
    dv_input = zeros(size(input));
    % TODO: BACKPROP CODE
    dv_input(input>=0) = dv_output;
    dv_input(input<0) = 0.01*dv_output;
end
