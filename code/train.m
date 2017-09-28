function [model, loss] = train(model,input,label,params,numIters)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = .01; end
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 128; end

% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);

for i = 1:numIters
	% TODO: Training code
    image_batch = input(:,:,:,floor(rand*(size(label,1)-batch_size)):floor(rand*(size(label,1)-batch_size))+batch_size-1);
    labels_batch = label(floor(rand*(size(label,1)-batch_size)):floor(rand*(size(label,1)-batch_size))+batch_size-1,:);
    % forward
    [final_layer_output,activations] = inference(model,image_batch);
    % loss
    [loss, dv_input_loss] = loss_crossentropy(final_layer_output, labels_batch, [], true);
    % backward
    [grad] = calc_gradient(model, image_batch, activations, dv_input_loss);
    % Update
    model = update_weights(model,grad,update_params);
end