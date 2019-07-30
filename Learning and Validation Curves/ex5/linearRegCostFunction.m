function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

for i = 1:m
  hx = sum(X(i,:) * theta);
  J = J + ((1 / (2 * m)) * ((hx - y(i)) ^ 2));
endfor

J = J + ((lambda / (2 * m)) * (sum((theta.^2))));

for i = 1:length(grad)
  for k = 1:m
    hx = sum(X(k,:) * theta);
    grad(i) = grad(i) + ((hx - y(k)) * X(k,i));
  endfor
  grad(i) = grad(i) + (lambda * theta(i));
  grad(i) = grad(i) / m;
endfor

grad(1) = grad(1) - ((lambda * theta(1)) / m);
J = J - (lambda / (2 * m)) * (theta(1)^2);






% =========================================================================

grad = grad(:);

end
