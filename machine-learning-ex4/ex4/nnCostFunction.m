function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

inputLayer = [ones(m,1) X];

#hidden layer hypothesis h(0)
hiddenLayer_hypo = inputLayer * Theta1';

hiddenLayer = sigmoid(hiddenLayer_hypo);

#adding first column as one to hypothesis(which will be used for gradient calculation) and hiddenlayer
hiddenLayer_hypo = [ones(m, 1) hiddenLayer_hypo];
hiddenLayer =  [ones(m, 1) hiddenLayer];

#output layer hypothesis h(0)
outputLayer_hypo = hiddenLayer * Theta2';

outputLayer = sigmoid(outputLayer_hypo);

#reducing  y = [5 ;4 ;3 ..] to y_extended = [0 0 0 0 1; 0 0 0 1 0; 0 0 1 0 0 ...]
y_extended = zeros(rows(y), max(y));
for i = 1:rows(y_extended)
  for j = 1:columns(y_extended)
    if (j == y(i))
        y_extended(i, j) = 1;
    endif
  endfor
endfor

temp = y_extended .* log(outputLayer) + (1 - y_extended) .* log(1 - outputLayer);

#sum(temp(:)) will return the sum of all elements in temp
J =  ( - 1 / m ) * sum(temp(:));

#Theta1(:, 2:end) will return all columns except first
reg_theta1 =  Theta1(: , 2:end) .* Theta1(: , 2:end);

reg_theta2 =  Theta2(: , 2:end) .* Theta2(: , 2:end);

reg_params = (lambda / (2 * m)) * (sum(reg_theta1(:)) + sum(reg_theta2(:)));

J = J + reg_params;


%%%%************ GRADIENT CALCULATION ************%%%%%%

delta3 = outputLayer - y_extended;

delta2 = (delta3 * Theta2) .* sigmoidGradient(hiddenLayer_hypo);
#removing the first column
delta2 = delta2( :, 2:end);

Theta2_grad =  (1 / m) * (Theta2_grad + delta3' * hiddenLayer);

Theta1_grad =  (1 / m) * (Theta1_grad + delta2' * inputLayer);

reg_term_theta1 = (lambda / m) * Theta1;
reg_term_theta2 = (lambda / m) * Theta2;

Theta1_grad_regularized = Theta1_grad + reg_term_theta1;

Theta2_grad_regularized = Theta2_grad + reg_term_theta2;

Theta1_grad = [Theta1_grad(:,1) Theta1_grad_regularized(:,2:end)];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad_regularized(:,2:end)];




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
