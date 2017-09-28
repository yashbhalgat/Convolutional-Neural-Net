function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

% TODO: FORWARD PROPAGATION CODE
for layer=1:num_layers
    % layer by layer forward propogation - backprop set as FALSE
    [activations{layer}, ~ , ~] = model.layers(layer).fwd_fn(input, model.layers(layer).params,model.layers(layer).hyper_params, false, []);
    
    % input to next layer
    input=activations{layer};
end

% output as final activation
output = activations{end};