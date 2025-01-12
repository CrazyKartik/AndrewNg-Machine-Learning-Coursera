function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    T0 = 0;
    T1 = 0;
    
    for i = 1:m
      T0 = T0 + (1 / m) * ( (theta(1,1) * X(i,1)) + (theta(2,1) * X(i,2)) - y(i,1) ) ;
      T1 = T1 + (1 / m) * ( ( (theta(1,1) * X(i,1)) + (theta(2,1) * X(i,2)) - y(i,1) ) * X(i,2) ) ;
    endfor

    th0 = theta(1,1) - alpha * T0 ;
    th1 = theta(2,1) - alpha * T1 ;

    theta(1,1) = th0;
    theta(2,1) = th1;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
