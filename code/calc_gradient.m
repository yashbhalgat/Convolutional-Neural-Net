function [grad] = calc_gradient(model, input, activations, dv_output)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just looping in reverse.

num_layers = numel(model.layers);
grad = cell(num_layers,1);

% TODO: Determine the gradient at each layer with weights to be updated
for layer=num_layers:-1:1
    % Going backward!
    % Idea taken from pseudocode for VGG
    if layer==1
        curr_input=input;
    else
        curr_input=activations{layer-1};
    end
    
    % SAME as inference code, just backprop is TRUE in this case
    [~, dv_input , grad{layer}] = model.layers(layer).fwd_fn(curr_input, model.layers(layer).params,model.layers(layer).hyper_params, true, dv_output);
    % pass to next layer
    dv_output = dv_input;
end