function MLP
	nombreArchivo = input('Escriba el archivo con el conjunto de entrenamiento: ', 's');
	itmax = input('Número de iteraciones máximas: ');
	alpha = input('Factor de aprendizaje: ');
    error = input('Error de iteración: ');
    porcentajeDatos = input('Porcentaje de datos a utilizar (1-100%): ');
	S1 = input('Ingrese el número de neuronas de la capa oculta: ');
	datos_sin_normalizar = load(nombreArchivo);
	disp(size(datos_sin_normalizar));

	[tablaTotal, minTabla, maxTabla, muTabla, sigmaTabla] = featureNormalize(datos_sin_normalizar);

	disp(size(tablaTotal));

    [Q, R] = size(tablaTotal);

	selectedRows =randperm(Q, ceil((porcentajeDatos/100.0)*Q));


	tablaTotal = tablaTotal(selectedRows, :);


	tabla = tablaTotal(:, 1);

	t = tablaTotal(:, 2);

	[Q, R] = size(tabla);

	disp(sprintf( 'Se usan %d filas.', Q));

	[Q, S2] = size(t);

	disp(sprintf('Arquitectura %d-%d-%d', R, S1, S2));

	fileErrores = fopen('errores.txt', 'w');

	W1 = rand(S1, R);
	b1 = rand(S1, 1);
	W2 = rand(S2, S1);
	b2 = rand(S2, 1);
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
			%disp(e);
            %%Calcular error
            err_it = err_it + (e.' * e) ;
		end
		it = it+1;
        %%Verificar condición de terminación por error.
        err_it = err_it / Q ;
        fprintf(fileErrores, '%f ', err_it);

        if error > err_it
            terminar_error = 1 ;
            break;
        end
	end
	fclose(fileErrores);
	disp(sprintf("Tiempo de ejecucion %f", toc()));
	a = zeros(1, Q);

	for m=1:Q
		a1 = logsig(W1*(tabla(:, m)) + b1);
		a2 = purelin(W2*a1 + b2);
		a(m) = a2;
	end

    if terminar_error > 0
        disp(sprintf('Aproximación exitosa en %d iteraciones.', it));
    else
        disp('Aproximación no exitosa');
    end
    disp(sprintf("Error de iteracion %f", err_it));

	%figure
	%hold on
	%plot(tabla, t, '* ');
	%plot(tabla, a, 'o ', 'color', 'r');
	%hold off
	tablaOriginal = featureDenormalize([tabla.', t.'],  minTabla, maxTabla, muTabla, sigmaTabla );
    tablaPredicha = featureDenormalize([tabla.', a.'], minTabla, maxTabla, muTabla, sigmaTabla );
    figure
	hold on
	plot(tablaOriginal(:, 1), tablaOriginal(:, 2), '* ');
	plot(tablaPredicha(:, 1), tablaPredicha(:, 2), 'o ', 'color', 'r');
	hold off
end
