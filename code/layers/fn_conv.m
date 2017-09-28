% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, padding for further work)
% params.W: filter_height x filter_width x filter_depth x num_filters
% params.b: num_filters x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)

[in_height,in_width,num_channels,batch_size] = size(input);
[filter_height,filter_width,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);
% TODO: FORWARD CODE
for indFilter = 1:num_filters
    for indImage = 1:batch_size
        % for each filter and each image
        for d = 1:num_channels
            output(:,:,indFilter,indImage) = output(:,:,indFilter,indImage) + conv2(input(:,:,d,indImage),params.W(:,:,d,indFilter),'valid');
        end
    end
    % And for every filter in the filter bank, add the bias
    output(:,:,indFilter,:) = output(:,:,indFilter,:)+params.b(indFilter);
end
    


dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    % First: computing dv_input
    dv_output_padded=padarray(dv_output,[filter_height-1,filter_width-1]);
    for h=1:in_height
        for w=1:in_width
            for c=1:num_channels
                % for each image
                for indImage=1:batch_size
                    dL_y=squeeze(dv_output_padded(h:h+filter_height-1,w:w+filter_width-1,:,indImage));
                    dv_input(h,w,c,indImage)=sum(sum(dot(squeeze(params.W(:,:,c,:)),dL_y)));
                end
            end
        end
    end
    
    %grad.W
    for h=1:filter_height
        for w=1:filter_width
            for d=1:filter_depth
                for indFilter=1:num_filters
                    dy_W=squeeze(input(filter_height-h+1:filter_height-h+out_height,filter_width-w+1:filter_width-w+out_width,d,:));
                    dL_y=squeeze(dv_output(:,:,indFilter,:));
                    grad.W(h,w,d,indFilter)=sum(sum(dot(dy_W,dL_y)));
                end
            end
        end
    end
    
    %grad.b
    for indFilter=1:num_filters
        grad.b(indFilter)=sum(sum(sum(sum(dv_output(:,:,indFilter,:)))));
    end
end