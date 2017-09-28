% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, filter_size)
% dv_output: same as output
% dv_input: same as input
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_pool(input, params, hyper_params, backprop, dv_output)
filter_size = hyper_params.filter_size;
stride = hyper_params.stride;
[~,~,num_channels,batch_size] = size(input);
assert(mod(size(input,1) - filter_size, stride)==0,...
	sprintf('Unsuitable stride and filter size'));
out_height = (size(input,1) - filter_size)/stride + 1;
out_width = (size(input,2) - filter_size)/stride + 1;
output = zeros(out_height,out_width,num_channels,batch_size);
% TODO: FORWARD CODE
% MAX POOLING
for indImage = 1:batch_size    
    for c = 1:num_channels
        for h = 1:out_height
            for w = 1:out_width
                h_start = (h-1)*stride+1;   h_end = h_start+filter_size-1;
                w_start = (w-1)*stride+1;   w_end = w_start+filter_size-1;
                output(h,w,c,indImage) = max(max(input(h_start:h_end,w_start:w_end,c,indImage)));
            end
        end
    end
end


dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    % dv_input
    for indImage = 1:batch_size    
        for c = 1:num_channels
            for h = 1:out_height
                for w = 1:out_width
                    h_start = (h-1)*stride+1;   h_end = h_start+filter_size-1;
                    w_start = (w-1)*stride+1;   w_end = w_start+filter_size-1;
                    input_mat = input(h_start:h_end,w_start:w_end,c,indImage);
                    [pos_r, pos_c] = find(input_mat==output(h,w,c,indImage)*ones(size(input_mat)));
                    dv_input(h_start+pos_r(1)-1,w_start+pos_c(1)-1,c,indImage) = dv_input(h_start+pos_r(1)-1,w_start+pos_c(1)-1,c,indImage)+dv_output(h,w,c,indImage);
                end
            end
        end
    end
%     dv_input = dv_input.*dv_output;
end