#include "mlp.h"

int main()
{
    neuron* mlp =  mlp_new(2, 3);
    //print_all_weight(mlp);
    gsl_matrix_float* input = gsl_matrix_float_alloc(1, 2);
    gsl_matrix_float_set(input, 0, 0, 1);
    gsl_matrix_float_set(input, 0, 1, 2);
    add_neuron(&mlp, 1, 3, 2);
    gsl_matrix_float* outputs = gsl_matrix_float_alloc(1, 2);
    forward_propogate(input, mlp, outputs);
    print_all_activations(mlp);
    gsl_matrix_float* error = gsl_matrix_float_alloc(1, 2);
    gsl_matrix_float_set(error, 0, 0, 0.1);
    gsl_matrix_float_set(error, 0, 1, 0.2);
    back_propogate(input, error, mlp, outputs);
    //print_all_weight(mlp);
    /*for (int i = 0; i < mlp->num_back; i++) {
        printf("mlp activations %d %f\r\n", i, gsl_matrix_float_get(mlp->activations,0, i));
    }*/
    /*gsl_matrix_float* a2 = gsl_matrix_float_alloc(2, 3);
    gsl_matrix_float_set(a2, 0, 0, 1);
    gsl_matrix_float_set(a2, 0, 1, 2);
    gsl_matrix_float_set(a2, 0, 2, 3);
    gsl_matrix_float_set(a2, 1, 0, 1);
    gsl_matrix_float_set(a2, 1, 1, 2);
    gsl_matrix_float_set(a2, 1, 2, 3);*/
    /*gsl_matrix_float* a3 = dot_product(2, 3, a1, a2);
    print_weight(1, 3, a3);
    gsl_matrix_float* a4 =  sigmoid_matrix(a3, 1, 3);
    print_weight(1, 3, a4);*/
    /*gsl_matrix_float* a3 = gsl_matrix_float_alloc(1, 2);
    gsl_matrix_float* a4 = gsl_matrix_float_alloc(1, 2);
    gsl_matrix_float* a6 = gsl_matrix_float_alloc(2, 1);
    gsl_matrix_float_set(a3, 0, 0, 1);
    gsl_matrix_float_set(a3, 0, 1, 3);
    gsl_matrix_float_transpose_memcpy(a6, a3);
    gsl_matrix_float_set(a4, 0, 0, 5);
    gsl_matrix_float_set(a4, 0, 1, 6);
    gsl_matrix_float* a5 = gsl_matrix_float_alloc(2, 2);
    gsl_blas_sgemm (CblasNoTrans, CblasNoTrans,
                1.0, a6, a4,
                0.0, a5);
    print_weight(2, 2, a5);*/
    gsl_matrix_float* input2 = gsl_matrix_float_alloc(1, 2);
    gsl_matrix_float_set(input2, 0, 0, 0.1);
    gsl_matrix_float_set(input2, 0, 1, 0.2);
    neuron* mlp2 = load_model();
    gsl_matrix_float* outputs2 = gsl_matrix_float_alloc(1, 2);
    forward_propogate(input2, mlp2, outputs2);
                

    return 0;
}