function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    T = zeros(1,size(X,2));
    th = zeros(1,size(X,2));
    for j = 1:size(X,2)
      for i = 1:m
        hx = 0;
        for k = 1:size(X,2)
          hx = hx + (theta(k,1) * X(i,k));
        endfor
        T(j) = T(j) + ((1 / m) * ( ( hx - y(i,1) ) * X(i,j) )) ;
      endfor
    endfor

    for i = 1:size(X,2)
      th(i) = theta(i,1) - alpha * T(i);
    endfor
    
    for i = 1:size(X,2)
      theta(i,1) = th(i);
    endfor





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
