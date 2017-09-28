%% Load data
load_MNIST_data

% Basic script to create a new network model

addpath layers;

accuracy_list=zeros(num_epoch,1);
train_loss_list=zeros(num_epoch,1);
test_loss_list=zeros(num_epoch,1);

l = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',8))
	init_layer('pool',struct('filter_size',4,'stride',2))
    init_layer('bn',struct('filter_depth',8))
	init_layer('relu',[])
%     init_layer('leaky_relu',[])
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',968,'num_out',10))
%     init_layer('linear',struct('num_in',400,'num_out',10))
	init_layer('softmax',[])];

model = init_model(l,[28 28 1],10,true);


% Example calls you might make:
% [output,~] = inference(model,input);
% [loss,~] = loss_euclidean(output,ground_truth,[],false);

%% Training
% paramters
num_epoch=128;
batch_size=100;
numIters=10;
num_batches=10;

for epoch=1:num_epoch
    epoch
    
    params=struct('learning_rate',.15,'weight_decay',0.0002,'batch_size',batch_size);
    
    [model, loss1] = train(model,train_data,train_label,params,numIters);
    train_loss_list(epoch)=loss1;
    %show result
    params1=struct('batch_size',batch_size);
    
    test_acc=zeros(num_batches,1);
    test_loss=zeros(num_batches,1);
    
    % measure test time
    tic
    
    parfor i=1:num_batches
        % Select non-random batches
        input_batch=test_data(:,:,:,(i-1)*batch_size+1:i*batch_size);
        label_batch=test_label((i-1)*batch_size+1:i*batch_size,:);
        [final_layer_output,~] = inference(model,input_batch);
        inferred_label=zeros(size(label_batch));
        for j=1:batch_size
            [~,inferred_label(j)]=max(final_layer_output(:,j));
        end
        curr_test_acc=inferred_label==label_batch;
        test_acc(i)=sum(curr_test_acc)/batch_size;
        [test_loss(i), ~] = loss_crossentropy(final_layer_output, label_batch, [], false);
    end
    accuracy = mean(test_acc);
    curr_loss = mean(test_loss);
    toc
    
    accuracy_list(epoch) = accuracy;
    test_loss_list(epoch) = curr_loss;
    
end