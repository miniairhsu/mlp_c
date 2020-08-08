#ifndef MLP_H
#define MLP_H
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_blas.h>
#include <time.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
struct Neuron {
    int index;
    int num_front;
    int num_back;
    gsl_matrix_float* inputs;
    gsl_matrix_float* pre_activations;
    gsl_matrix_float* activations;
    gsl_matrix_float* weight;
    gsl_matrix_float* weight_derivative;
    gsl_matrix_float* bias;
    struct Neuron *next;
    struct Neuron *prev;
};

typedef struct Neuron neuron; 
neuron* mlp_new(int num_front, int num_back);

void add_neuron(neuron** head, int index, int num_front, int num_back);

void init_weight(int row, int col, gsl_matrix_float* weight);

void dot_product(int num_front, int num_back, gsl_matrix_float* m1, gsl_matrix_float* m2, gsl_matrix_float* output);

void dot_product_bias(int num_front, int num_back, gsl_matrix_float* m1, gsl_matrix_float* m2, gsl_matrix_float* bias, 
    gsl_matrix_float* output);

gsl_matrix_float* sigmoid_matrix(gsl_matrix_float* net_inputs, int row, int col); 

void sigmoid_matrix_derivative(gsl_matrix_float* net_inputs, int row, int col, gsl_matrix_float* output);

void forward_propogate(gsl_matrix_float* input, neuron* mlp, gsl_matrix_float* outputs);

void back_propogate(gsl_matrix_float* inputs, gsl_matrix_float* error, neuron* mlp, gsl_matrix_float* outputs);

void print_weight(int row, int col, gsl_matrix_float* weight);

void print_all_activations(neuron* head);

void print_all_weight(neuron* head);

neuron* load_model();

#endif