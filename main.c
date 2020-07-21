#include "mlp.h"

int main()
{
    neuron* mlp =  mlp_new(2, 3);
    //print_all_weight(mlp);
    gsl_matrix_float* a1 = gsl_matrix_float_alloc(1, 2);
    gsl_matrix_float_set(a1, 0, 0, 1);
    gsl_matrix_float_set(a1, 0, 1, 2);
    add_neuron(&mlp, 1, 3, 2);
    gsl_matrix_float* outputs = gsl_matrix_float_alloc(1, 2);
    forward_propogate(a1, mlp, outputs);
    print_all_activations(mlp);
    print_all_weight(mlp);
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
    gsl_matrix_float* a3 = gsl_matrix_float_alloc(2, 2);
    gsl_matrix_float* a4 = gsl_matrix_float_alloc(2, 2);
    gsl_matrix_float_set(a3, 0, 0, 1);
    gsl_matrix_float_set(a3, 0, 1, 2);
    gsl_matrix_float_set(a3, 1, 0, 3);
    gsl_matrix_float_set(a3, 1, 1, 4);

    gsl_matrix_float_set(a4, 0, 0, 5);
    gsl_matrix_float_set(a4, 0, 1, 6);
    gsl_matrix_float_set(a4, 1, 0, 7);
    gsl_matrix_float_set(a4, 1, 1, 8);
    gsl_matrix_float* a5 = gsl_matrix_float_alloc(2, 2);
    gsl_matrix_float_mul_elements(a3, a4);
    print_weight(2, 2, a3);
                

    return 0;
}