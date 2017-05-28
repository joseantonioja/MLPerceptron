function MLP
	nombreArchivoTabla = input('Escriba el archivo con los datos del conjunto de entrenamiento: ', 's');
	nombreArchivoTargets =  input('Escriba el archivo con los targets: ', 's');
	itmax = input('Número de iteraciones máximas: ');
	alpha = input('Factor de aprendizaje: ');
    error = input('Error de iteración: ');  
	tabla = load(nombreArchivoTabla);
	t= load(nombreArchivoTargets);
	S1 = 2
	[Q, R] = size(tabla);
	[Q, S2] = size(t);
	disp(sprintf('Arquitectura %d-%d-%d', R, S1, S2))
	
	W1 = [-0.27; -0.41];
	b1 = [-0.48; -0.13];
	W2 = [0.09, -0.17];
	b2 = [0.48];
	
	%W1 = rand(S1, R)
	%b1 = rand(S1, 1)
	%W2 = rand(S2, S1)
	%b2 = rand(S2, 1)
	it = 0;
	tabla = tabla.';
	t = t.';
	tic()
    
    terminar_error = 0 ;
	while it < itmax
        err_it = 0 ;
		for m = 1:Q
			%%Propagacion hacia delante del m-esimo elemento
			a1 = logsig(W1*(tabla(:, m)) + b1);
			a2 = purelin(W2*a1 + b2);
			%%Calculo de las sensibilidades
			e = t(: , m) - a2;
			s2 = -2*1*e;
			F1 = diag( (1-a1).*a1 );
			s1 = F1*(W2.')*s2;
			%%Actualizacion de pesos y bias
			W2 = W2 - alpha*s2*(a1.');
			b2 = b2 - alpha*s2;
			W1 = W1 - alpha*s1*(tabla(:, m).');
			b1 = b1 - alpha*s1;
            %%Calcular error
            err_it = err_it + (transpose(e)*e) ;
		end
		it = it+1;
        %%Verificar condición de terminación por error.
        err_it = err_it / Q ;
        if error > err_it 
            terminar_error = 1 ;
            break;
        end 
	end
	disp(toc())
	a = zeros(1, Q);
	for m=1:Q
		a1 = logsig(W1*(tabla(:, m)) + b1);
		a2 = purelin(W2*a1 + b2);
		a(m) = a2;
	end
	disp(a);
    
    if terminar_error > 0
        disp('Aproximación exitosa.');
    else
        disp('Aproximación no exitosa');
    end
    disp(err_it);
    
	figure
	hold on
	plot(tabla, t, '* ');
	plot(tabla, a, 'r-');
	hold off
end