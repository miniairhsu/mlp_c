#include "mlp.h"

neuron* mlp_new(int num_front, int num_back)
{
    neuron* m = (neuron *)malloc(sizeof(neuron));
    m->index = 0;
    m->num_front = num_front;
    m->num_back = num_back;
    m->inputs = gsl_matrix_float_alloc(1, num_front);
    m->weight = gsl_matrix_float_alloc(num_front, num_back);
    m->activations = gsl_matrix_float_alloc(1, num_back);
    init_weight(num_front, num_back, m->weight);
    print_weight(num_front, num_back, m->weight);
    m->next = NULL;
    m->prev = NULL;
}

void add_neuron(neuron** head, int index, int num_front, int num_back)
{
    neuron* n = (neuron *)malloc(sizeof(neuron));
    neuron* last = *head;
    n->index = index;
    n->num_front = num_front;
    n->num_back = num_back;
    n->inputs = gsl_matrix_float_alloc(1, num_front);
    n->weight = gsl_matrix_float_alloc(num_front, num_back);
    n->activations = gsl_matrix_float_alloc(1, num_back);
    init_weight(num_front, num_back, n->weight);
    print_weight(num_front, num_back, n->weight);
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

void init_weight(int row, int col, gsl_matrix_float* weight)
{   
    srand(time(0));
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            gsl_matrix_float_set(weight, i, j, (float)rand()/RAND_MAX);
    return;
}

void print_weight(int row, int col, gsl_matrix_float* weight)
{
    for (int i = 0; i < row; i++) 
        for (int j = 0; j < col; j++)
            printf("weight(%d,%d) = %g\n", i, j,
              gsl_matrix_float_get(weight, i, j));
}

void print_all_weight(neuron* head)
{
    neuron* temp = head;
    neuron* temp_reverse;
    while (temp != NULL) {
        printf("weight index %d\r\n", temp->index);
        print_weight(temp->num_front, temp->num_back, temp->weight);
        temp_reverse = temp;
        temp = temp->next;
    }
    free(temp);
    /*while (temp_reverse != NULL) {
        printf("weight reverse index %d\r\n", temp_reverse->index);
        print_weight(temp_reverse->num_front, temp_reverse->num_back, temp_reverse->weight);
        temp_reverse = temp_reverse->prev;
    }*/
    free(temp_reverse);
}

void print_all_activations(neuron* head)
{
    neuron* temp = head;
    while (temp != NULL) {
        printf("activation index %d\r\n", temp->index);
        print_weight(1, temp->num_back, temp->activations);
        temp = temp->next;
    }
    free(temp);
}

gsl_matrix_float* dot_product(int num_front, int num_back, gsl_matrix_float* m1, gsl_matrix_float* m2)
{
    gsl_matrix_float* outputs = gsl_matrix_float_alloc(1, num_back);
    //for (int i = 0; i < num_inputs; i++) {
        gsl_vector_float_view row_m1 = gsl_matrix_float_row(m1, 0);
        for (int j = 0; j < num_back; j++) {
            float result = 0;
            gsl_vector_float_view col_m2 = gsl_matrix_float_column(m2, j);
            gsl_blas_sdot(&row_m1.vector, &col_m2.vector, &result);
            gsl_matrix_float_set(outputs, 0, j, result);
        }
    //}
    return outputs;
}

gsl_matrix_float* sigmoid_matrix(gsl_matrix_float* net_inputs, int row, int col)
{
    gsl_matrix_float* output = gsl_matrix_float_alloc(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float result = 1 / (1 + expf(-1 * gsl_matrix_float_get(net_inputs, i, j)));
            gsl_matrix_float_set(output, i, j, result);
        }
    }
    return output;
}

gsl_matrix_float* sigmoid_matrix_derivative(gsl_matrix_float* net_inputs, int row, int col)
{
    gsl_matrix_float* output = gsl_matrix_float_alloc(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float result = gsl_matrix_float_get(net_inputs, i, j);
            result = result / (1 - result);
            gsl_matrix_float_set(output, i, j, result);
        }
    }
    return output;
}

void forward_propogate(gsl_matrix_float* input, neuron* mlp, gsl_matrix_float* outputs)
{
    neuron* temp = mlp;
    mlp->inputs = input;
    temp->inputs = input;
    while (temp != NULL) {
        gsl_matrix_float* net_inputs = dot_product(temp->num_front, temp->num_back, temp->inputs, temp->weight);
        gsl_matrix_float* sigmoid = sigmoid_matrix(net_inputs, 1, temp->num_back);
        gsl_matrix_float_memcpy(temp->activations, sigmoid);
        if (temp->next != NULL) {
           gsl_matrix_float_memcpy((temp->next)->inputs, temp->activations);
        }
        gsl_matrix_float_free(net_inputs);
        gsl_matrix_float_free(sigmoid);
        temp = temp->next;
    }
    return;
}

void back_propogate(gsl_matrix_float* inputs, gsl_matrix_float* error, neuron* mlp, gsl_matrix_float* outputs)
{
    neuron* temp = mlp;
    neuron* temp_reverse = temp;
    //temp_reverse->activations = inputs;
    while (temp != NULL) {
        temp_reverse = temp;
        temp = temp->next;
    }
    while (temp_reverse != NULL) {
        gsl_matrix_float* drivatives = sigmoid_matrix_derivative(temp_reverse->activations, 1, temp_reverse->num_back);
        gsl_matrix_float_mul_elements(drivatives, error);
        print_weight(1, temp_reverse->num_back, drivatives);
        gsl_matrix_float* temp_matrix = gsl_matrix_float_alloc(temp_reverse->num_front, 1);
        gsl_matrix_float* temp_result = gsl_matrix_float_alloc(temp_reverse->num_front, temp_reverse->num_back);
        gsl_matrix_float_transpose_memcpy(temp_matrix, (temp_reverse)->inputs);
        gsl_blas_sgemm (CblasNoTrans, CblasNoTrans,
                    1.0, temp_matrix, drivatives,
                    0.0, temp_result);
        //print_weight(temp_reverse->num_front, temp_reverse->num_back, temp_result);
        error = dot_product(temp_reverse->num_front, temp_reverse->num_back, drivatives, temp_reverse->weight);
        gsl_matrix_float_free(temp_matrix);
        gsl_matrix_float_free(temp_result);
        /*gsl_matrix_float* temp_result = gsl_matrix_float_alloc(temp_reverse->num_front, temp_reverse->num_back);
        gsl_blas_sgemm (CblasNoTrans, CblasNoTrans,
                    1.0, temp_matrix, drivatives,
                    0.0, temp_result);*/
       
        /*gsl_matrix_float* temp_matrix = gsl_matrix_float_alloc(temp_reverse->num_back, temp_reverse->num_front);
        gsl_matrix_float* temp_result = gsl_matrix_float_alloc(temp_reverse->num_front, temp_reverse->num_back);
        gsl_matrix_float_transpose_memcpy(temp_matrix, (temp_reverse->prev)->activations);
        print_weight(temp_reverse->num_back, temp_reverse->num_front, temp_matrix);
        gsl_blas_sgemm (CblasNoTrans, CblasNoTrans,
                    1.0, temp_matrix, drivatives,
                    0.0, temp_result);
        print_weight(temp_reverse->num_front, temp_reverse->num_back, temp_result);
        gsl_matrix_float_free(temp_matrix);
        gsl_matrix_float_free(temp_result);*/
        temp_reverse = temp_reverse->prev;

    }
    free(temp_reverse);
    free(temp);
}