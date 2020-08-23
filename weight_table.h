#ifndef WEIGHT_TABLE_H
#define WEIGHT_TABLE_H
#define NUM_LAYER 3
#define SIGMOID 0
#define RELU 1
#define SOFTMAX 2
int* load_weight_size();
float* load_weights();
int* load_activation_type();
#endif