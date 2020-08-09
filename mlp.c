#include "mlp.h"
#include "weight_table.h"

neuron *mlp_new(int num_front, int num_back)
{
	neuron *m = (neuron *)malloc(sizeof(neuron));
	m->index = 0;
	m->num_front = num_front;
	m->num_back = num_back;
	m->inputs = gsl_matrix_float_alloc(1, num_front);
	m->weight = gsl_matrix_float_alloc(num_front, num_back);
	m->weight_derivative = gsl_matrix_float_alloc(num_front, num_back);
	m->pre_activations = gsl_matrix_float_alloc(1, num_back);
	m->activations = gsl_matrix_float_alloc(1, num_back);
	m->bias = gsl_matrix_float_alloc(1, num_back);
	set_activation(m, sigmoid_matrix);
	init_weight(num_front, num_back, m->weight);
	// print_weight(num_front, num_back, m->weight);
	m->next = NULL;
	m->prev = NULL;
}

void add_neuron(neuron **head, int index, int num_front, int num_back)
{
	neuron *n = (neuron *)malloc(sizeof(neuron));
	neuron *last = *head;
	n->index = index;
	n->num_front = num_front;
	n->num_back = num_back;
	n->inputs = gsl_matrix_float_alloc(1, num_front);
	n->weight = gsl_matrix_float_alloc(num_front, num_back);
	n->weight_derivative = gsl_matrix_float_alloc(num_front, num_back);
	n->pre_activations = gsl_matrix_float_alloc(1, num_back);
	n->activations = gsl_matrix_float_alloc(1, num_back);
	n->bias = gsl_matrix_float_alloc(1, num_back);
	set_activation(n, sigmoid_matrix);
	init_weight(num_front, num_back, n->weight);
	// print_weight(num_front, num_back, n->weight);
	n->next = NULL;
	if (head == NULL) {
		*head = n;
		return;
	}
	while (last->next != NULL) {
		last = last->next;
	}
	last->next = n;
	n->prev = last;
	last = last->next;
	return;
}

void set_activation(neuron* mlp, gsl_matrix_float* (*activation) (gsl_matrix_float*, int, int))
{
	mlp->activation = activation;
}

void init_weight(int row, int col, gsl_matrix_float *weight)
{
	srand(time(0));
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			gsl_matrix_float_set(weight, i, j,
					     (float)rand() / RAND_MAX);
	return;
}

void set_weight(int num_row, int num_col, neuron *mlp, float *weight_table)
{
	for (int i = 0; i < num_row; i++) {
		for (int j = 0; j < num_col; j++) {
			gsl_matrix_float_set(mlp->weight, i, j,
					     weight_table[i * num_col + j]);
		}
	}
}

void set_bias(int num_col, neuron *mlp, float *weight_table)
{
	for (int j = 0; j < num_col; j++) {
		gsl_matrix_float_set(mlp->bias, 0, j, weight_table[j]);
	}
}

void print_weight(int row, int col, gsl_matrix_float *weight)
{
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			printf("weight(%d,%d) = %g\n", i, j,
			       gsl_matrix_float_get(weight, i, j));
}

void print_all_weight(neuron *head)
{
	neuron *temp = head;
	neuron *temp_reverse;
	while (temp != NULL) {
		printf("weight index %d\r\n", temp->index);
		print_weight(temp->num_front, temp->num_back, temp->weight);
		temp_reverse = temp;
		temp = temp->next;
	}
	free(temp);
	/*while (temp_reverse != NULL) {
	    printf("weight reverse index %d\r\n", temp_reverse->index);
	    print_weight(temp_reverse->num_front, temp_reverse->num_back,
	temp_reverse->weight); temp_reverse = temp_reverse->prev;
	}*/
	free(temp_reverse);
}

void print_all_activations(neuron *head)
{
	neuron *temp = head;
	while (temp != NULL) {
		printf("activation index %d\r\n", temp->index);
		print_weight(1, temp->num_back, temp->activations);
		temp = temp->next;
	}
	free(temp);
}

void dot_product(int num_front, int num_back, gsl_matrix_float *m1,
		 gsl_matrix_float *m2, gsl_matrix_float *output)
{
	// gsl_matrix_float* outputs = gsl_matrix_float_alloc(1, num_back);
	// for (int i = 0; i < num_inputs; i++) {
	gsl_vector_float_view row_m1 = gsl_matrix_float_row(m1, 0);
	for (int j = 0; j < num_back; j++) {
		float result = 0;
		gsl_vector_float_view col_m2 = gsl_matrix_float_column(m2, j);
		gsl_blas_sdot(&row_m1.vector, &col_m2.vector, &result);
		gsl_matrix_float_set(output, 0, j, result);
	}
	//}
	// return outputs;
}

void dot_product_bias(int num_front, int num_back, gsl_matrix_float *m1,
		      gsl_matrix_float *m2, gsl_matrix_float *bias,
		      gsl_matrix_float *output)
{
	// gsl_matrix_float* outputs = gsl_matrix_float_alloc(1, num_back);
	// for (int i = 0; i < num_inputs; i++) {
	gsl_vector_float_view row_m1 = gsl_matrix_float_row(m1, 0);
	for (int j = 0; j < num_back; j++) {
		float result = 0;
		gsl_vector_float_view col_m2 = gsl_matrix_float_column(m2, j);
		gsl_blas_sdot(&row_m1.vector, &col_m2.vector, &result);
		result += gsl_matrix_float_get(bias, 0, j);
		gsl_matrix_float_set(output, 0, j, result);
	}
	//}
	// return outputs;
}

gsl_matrix_float *sigmoid_matrix(gsl_matrix_float *net_inputs, int row, int col)
{
	gsl_matrix_float *output = gsl_matrix_float_alloc(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float result = 1
				       / (1
					  + expf(-1
						 * gsl_matrix_float_get(
							 net_inputs, i, j)));
			gsl_matrix_float_set(output, i, j, result);
		}
	}
	return output;
}

void sigmoid_matrix_derivative(gsl_matrix_float *net_inputs, int row, int col,
			       gsl_matrix_float *output)
{
	// gsl_matrix_float* output = gsl_matrix_float_alloc(row, col);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float result = gsl_matrix_float_get(net_inputs, i, j);
			result = result / (1 - result);
			gsl_matrix_float_set(output, i, j, result);
		}
	}
	// return output;
}

void forward_propogate(gsl_matrix_float *input, neuron *mlp,
		       gsl_matrix_float *outputs)
{
	neuron *temp = mlp;
	mlp->inputs = input;
	temp->inputs = input;
	gsl_matrix_float *net_inputs =
		gsl_matrix_float_alloc(1, temp->num_back);
	while (temp != NULL) {
		printf("forward progation layer %d inputs\r\n", temp->index);
		print_weight(1, temp->num_front, temp->inputs);
		dot_product_bias(temp->num_front, temp->num_back, temp->inputs,
				 temp->weight, temp->bias, net_inputs);
		gsl_matrix_float_memcpy(temp->pre_activations, net_inputs);
		printf("forward progation pre activation\r\n");
		print_weight(1, temp->num_back, temp->pre_activations);
		gsl_matrix_float *sigmoid =
			sigmoid_matrix(net_inputs, 1, temp->num_back);
		gsl_matrix_float_memcpy(temp->activations, sigmoid);
		printf("forward progation activation\r\n");
		print_weight(1, temp->num_back, temp->activations);
		if (temp->next != NULL) {
			gsl_matrix_float_memcpy((temp->next)->inputs,
						temp->activations);
			net_inputs = gsl_matrix_float_calloc(
				1, (temp->next)->num_back);
		}
		gsl_matrix_float_free(sigmoid);
		temp = temp->next;
	}
	gsl_matrix_float_free(net_inputs);
	return;
}

void back_propogate(gsl_matrix_float *inputs, gsl_matrix_float *error,
		    neuron *mlp, gsl_matrix_float *outputs)
{
	neuron *temp = mlp;
	neuron *temp_reverse = temp;
	while (temp != NULL) {
		temp_reverse = temp;
		temp = temp->next;
	}
	gsl_matrix_float *errors =
		gsl_matrix_float_alloc(1, temp_reverse->num_back);
	gsl_matrix_float *drivatives =
		gsl_matrix_float_alloc(1, temp_reverse->num_back);
	gsl_matrix_float *input_trans =
		gsl_matrix_float_alloc(temp_reverse->num_front, 1);
	gsl_matrix_float *temp_result = gsl_matrix_float_alloc(
		temp_reverse->num_front, temp_reverse->num_back);
	gsl_matrix_float *weight_trans = gsl_matrix_float_calloc(
		temp_reverse->num_back, temp_reverse->num_front);
	gsl_matrix_float_memcpy(errors, error);
	while (temp_reverse != NULL) {
		sigmoid_matrix_derivative(temp_reverse->activations, 1,
					  temp_reverse->num_back, drivatives);
		printf("backward progation layer %d sigmoid derivative(pre_activation)\r\n",
		       temp_reverse->index);
		print_weight(1, temp_reverse->num_back, drivatives);
		gsl_matrix_float_mul_elements(drivatives, errors);
		gsl_matrix_float_transpose_memcpy(input_trans,
						  temp_reverse->inputs);
		gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, input_trans,
			       drivatives, 0.0, temp_result);
		printf("dW / dE\r\n");
		gsl_matrix_float_memcpy(temp_reverse->weight_derivative,
					temp_result);
		print_weight(temp_reverse->num_front, temp_reverse->num_back,
			     temp_reverse->weight_derivative);
		if (temp_reverse->prev != NULL) {
			errors = gsl_matrix_float_calloc(
				1, temp_reverse->num_front);
			gsl_matrix_float_transpose_memcpy(
				weight_trans, (temp_reverse)->weight);
			printf("weight . sigmoid derivative(pre_activation)\r\n");
			dot_product(temp_reverse->num_front,
				    temp_reverse->num_back, drivatives,
				    weight_trans, errors);
			print_weight(1, temp_reverse->num_front, errors);
			weight_trans = gsl_matrix_float_calloc(
				temp_reverse->num_back,
				temp_reverse->num_front);
		}
		if (temp_reverse->prev != NULL) {
			temp_reverse = temp_reverse->prev;
			drivatives = gsl_matrix_float_calloc(
				1, temp_reverse->num_back);
			input_trans = gsl_matrix_float_calloc(
				temp_reverse->num_front, 1);
			temp_result =
				gsl_matrix_float_calloc(temp_reverse->num_front,
							temp_reverse->num_back);
		} else {
			break;
		}
	}
	gsl_matrix_float_free(drivatives);
	gsl_matrix_float_free(errors);
	gsl_matrix_float_free(input_trans);
	gsl_matrix_float_free(temp_result);
	gsl_matrix_float_free(weight_trans);
	free(temp_reverse);
	free(temp);
}

neuron *load_model()
{
	int index = 0;
	int layer_size = NUM_LAYER;
	int *weight_size = load_weight_size();
	float *weights = load_weights();
	neuron *mlp = mlp_new(weight_size[0], weight_size[1]);
	neuron *mlp_ret = mlp;
	int *weight_size_temp = weight_size;
	float *weights_temp = weights;
	index++;
	int weight_index = 0;
	int bias_index = weight_size[0] * weight_size[1];
	for (int i = 0; i < layer_size; i++) {
		printf("Layer %d\r\n", i);
		set_weight(weight_size[i * 3], weight_size[i * 3 + 1], mlp,
			   &weights[weight_index]);
		set_bias(weight_size[i * 3 + 2], mlp, &weights[bias_index]);
		weight_index = weight_index
			       + weight_size[i * 3] * weight_size[i * 3 + 1]
			       + weight_size[i * 3 + 2];
		print_weight(weight_size[i * 3], weight_size[i * 3 + 1],
			     mlp->weight);
		print_weight(1, weight_size[i * 3 + 2], mlp->bias);
		if (i < (layer_size - 1)) {
			weight_size_temp = (&weight_size[(i + 1) * 3]);
			bias_index =
				weight_index
				+ weight_size_temp[0] * weight_size_temp[1];
			add_neuron(&mlp, index, weight_size_temp[0],
				   weight_size_temp[1]);
			mlp = mlp->next;
		}
		index++;
	}
	return mlp_ret;
}