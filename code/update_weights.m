function updated_model = update_weights(model,grad,hyper_params)

num_layers = length(grad);
a = hyper_params.learning_rate;
lmda = hyper_params.weight_decay;
updated_model = model;

% TODO: Update the weights of each layer in your model based on the calculated gradients
for layer=1:num_layers
    % Momentum NOT used
    updated_model.layers(layer).params.W=(1-lmda).*updated_model.layers(layer).params.W-a.*grad{layer}.W;
    updated_model.layers(layer).params.b=(1-lmda).*updated_model.layers(layer).params.b-a.*grad{layer}.b;    
end