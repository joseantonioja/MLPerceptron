function [X] = featureDenormalize(X_norm, minX, maxX, mu, sigma)
%FEATUREDENORMALIZE Denormalizes the features in X
%   FEATUREDENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when

X = bsxfun(@times, X_norm, sigma);
X = bsxfun(@plus, X, mu);
X = bsxfun(@times, X, maxX - minX);
X = bsxfun(@plus, X, minX);
% ============================================================

end
