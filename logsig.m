function g = logsig(z)
%logsig(z) - Evalua la funcion log-sigmoid de z
	g = 1.0 ./ (1.0 + exp(-z));
end
