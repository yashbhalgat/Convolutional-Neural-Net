% Load the MNIST data into the MATLAB workspace, and set a flag so that the
% data does not have to be reloaded everytime you want to test something.
addpath ../data;

if ~exist('MNIST_loaded')
	% Load training data
	train_data = load_MNIST_images('../data/train-images.idx3-ubyte');
	train_data = reshape(train_data,28,28,1,[]);
	train_label = load_MNIST_labels('../data/train-labels.idx1-ubyte');
	train_label(train_label == 0) = 10; % Remap 0 to 10

	% Load testing data
	test_data = load_MNIST_images('../data/t10k-images.idx3-ubyte');
	test_data = reshape(test_data,28,28,1,[]);
	test_label = load_MNIST_labels('../data/t10k-labels.idx1-ubyte');
	test_label(test_label == 0) = 10; % Remap 0 to 10

	MNIST_loaded = true;
end
