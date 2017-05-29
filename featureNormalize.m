function [X_norm, minX, maxX, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when

minX = min(X);
maxX = max(X);
X_norm = bsxfun(@rdivide, X - minX, maxX - minX);
mu = mean(X_norm);
sigma = mean(X_norm);
X_norm = bsxfun(@minus, X_norm, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

% ============================================================

end
