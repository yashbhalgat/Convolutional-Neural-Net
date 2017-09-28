% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
    loss = 0;
    [num_nodes,batch_size] = size(input);
    % generate a one-hot matrix for labels
    one_hot = zeros(num_nodes, batch_size);
    for indImage = 1:batch_size
        one_hot(labels(indImage),indImage) = 1;
    end
    loss = -sum(dot(log(input),one_hot));
    loss = loss/batch_size;

dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    dv_input = -(one_hot./input);
    dv_input = dv_input/batch_size;
end
