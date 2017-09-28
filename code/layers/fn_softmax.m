% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);
output = zeros(num_classes, batch_size);
% TODO: FORWARD CODE
output = exp(input);
denominator = sum(output,1);
output = output./repmat(denominator,num_classes,1);


dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    for image_number = 1:batch_size
        dv_y = dv_output(:,image_number);
        y = output(:,image_number);

        DyDx=zeros(num_classes,num_classes);
        for i=1:num_classes %y index
            for j=1:num_classes %x index
                if i~=j
                    DyDx(i,j) = -y(i)*y(j);
                else
                    DyDx(i,j) = y(i)*(1-y(j));
                end
            end
        end
        
        % DyDx is the Jacobian matrix, so matrix multiplication
        dv_input(:,image_number) = DyDx*dv_y;
    end
end