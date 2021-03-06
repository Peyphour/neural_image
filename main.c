#include <fann.h>
#include <floatfann.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "graphics.c"

typedef struct {
    int mag_number;
    int num_items;
    int num_rows;
    int num_cols;
} idx_image_format;

typedef struct {
    int mag_number;
    int num_items;
} idx_label_format;

int byteswap32(int x) {
    x = (x & 0x0000FFFF) << 16 | (x & 0xFFFF0000) >> 16;
    x = (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8;
    return x;
}

char *convert_int_to_string(int num) {
    switch(num) {
        case 0:
            return "1 0 0 0 0 0 0 0 0 0";
        case 1:
            return "0 1 0 0 0 0 0 0 0 0";
        case 2:
            return "0 0 1 0 0 0 0 0 0 0";
        case 3:
            return "0 0 0 1 0 0 0 0 0 0";
        case 4:
            return "0 0 0 0 1 0 0 0 0 0";
        case 5:
            return "0 0 0 0 0 1 0 0 0 0";
        case 6:
            return "0 0 0 0 0 0 1 0 0 0";
        case 7:
            return "0 0 0 0 0 0 0 1 0 0";
        case 8:
            return "0 0 0 0 0 0 0 0 1 0";
        case 9:
            return "0 0 0 0 0 0 0 0 0 1";
        default:
            return "0 0 0 0 0 0 0 0 0 0";
    }
}

void init_SDL() {
	return init_graphics(50, 50, "Neural Network");
}

unsigned char **get_image_data(FILE *filename, idx_image_format *image_format) {

    unsigned char **data;
    int i, j;

    fread(image_format, sizeof(idx_image_format), 1, filename);

    image_format->mag_number = byteswap32(image_format->mag_number);
    image_format->num_items = byteswap32(image_format->num_items);
    image_format->num_rows = byteswap32(image_format->num_rows);
    image_format->num_cols = byteswap32(image_format->num_cols);

    data = (unsigned char **) malloc(image_format->num_items * sizeof(unsigned char *));
    for(i = 0; i < image_format->num_items; i++) {
        data[i] = (unsigned char *) malloc(image_format->num_cols * image_format->num_rows * sizeof(unsigned char));
        for(j = 0; j < image_format->num_cols * image_format->num_rows; j++)
            fscanf(filename, "%c", &data[i][j]);
    }

    fclose(filename);

    return data;
}

unsigned char *get_label_data(FILE *filename) {

    idx_label_format label_format;
    unsigned char *labels;
    int i;

    fread(&label_format, sizeof(idx_label_format), 1, filename);

    label_format.num_items = byteswap32(label_format.num_items);
    label_format.mag_number = byteswap32(label_format.mag_number);

    labels = (unsigned char *) malloc(label_format.num_items * sizeof(unsigned char*));
    for(i = 0; i < label_format.num_items; i++) {
        fscanf(filename, "%c", &labels[i]);
    }

    fclose(filename);

    return labels;
}

void print_to_ann_format(FILE *output, idx_image_format image_format, unsigned char **data, unsigned char *labels) {

    int i, j;

    fprintf(output, "%d %d %d\n", image_format.num_items, image_format.num_cols * image_format.num_rows, 10);
    for(i = 0; i < image_format.num_items; i++) {
        for(j = 0; j < image_format.num_rows * image_format.num_cols; j++) {
            fprintf(output, "%f", (float)data[i][j]/255);
            if(j != image_format.num_rows * image_format.num_cols - 1)
                fprintf(output, " ");
        }
        fprintf(output, "\n");
        fprintf(output, "%s\n", convert_int_to_string(labels[i]));
    }

    fclose(output);
}

void translate_idx_to_fann() {
    FILE *image, *label, *output;
    idx_image_format image_format;

    int i, j, k;

    unsigned char **data, *labels;

    image = fopen("./train-images.idx3-ubyte", "r");
    label = fopen("./train-labels.idx1-ubyte", "r");
    output = fopen("./image.data", "w+");

    data = get_image_data(image, &image_format);

    labels = get_label_data(label);

    print_to_ann_format(output, image_format, data, labels);
}


fann_type *from_char_to_fann_type(unsigned char *data, int size) {
    fann_type *return_data = (fann_type*) malloc(size * sizeof(fann_type));
    int i;

    for(i = 0; i < size; i++) {
        return_data[i] = (fann_type)data[i]/255;
    }
    return return_data;
}

int max_tab(fann_type *in, int size) {
    fann_type max = in[0];
    int max_index = 0, i;

    for(i = 1; i < size; i++) {
        if(in[i] >= max) {
            max = in[i];
            max_index = i;
        }
    }
    return max_index;
}

void run_tests() {
    fann_type *calc_out, *input;

    idx_image_format image_format;
    FILE *test_set;
    FILE *label_set;

    unsigned char **test_data, *test_labels;
    int i, j, k, guessed_num;
	POINT p;

	init_SDL();

    test_set = fopen("./t10k-images.idx3-ubyte", "r");
    label_set = fopen("./t10k-labels.idx1-ubyte", "r");

    struct fann *ann = fann_create_from_file("image_float.net");

    test_data = get_image_data(test_set, &image_format);
    test_labels = get_label_data(label_set);

    for(i = 0; i < image_format.num_items; i++) {
        calc_out = fann_run(ann, from_char_to_fann_type(test_data[i], image_format.num_rows*image_format.num_cols));
        fill_screen(noir);
        for(j = 0; j < image_format.num_rows; j++) {
            for(k = 0; k < image_format.num_cols; k++) {
                //printf("%c", test_data[i][j*image_format.num_cols + k]>20?'.':' ');
                p.y = 50-j;
                p.x = k;
                if(test_data[i][j*image_format.num_cols + k] > 20)
                	draw_pixel(p, blanc);
                if(k != image_format.num_cols - 1)
                    printf(" ");
            }
            printf("\n");
        }
        guessed_num = max_tab(calc_out, 10);
        printf("Guess : %d (real : %d) ", guessed_num, test_labels[i]);
        if(guessed_num != test_labels[i]) {
            printf("probability of the  correct guess was %f, the probability of the guessed one was %f", calc_out[test_labels[i]], calc_out[guessed_num]);
        }
        getchar();
    }

    fann_destroy(ann);
}

int main(int argc, char *argv[]) {

    const unsigned int num_input = 784;
    const unsigned int num_output = 10;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 30;
    const float desired_error = (const float) 0.012;
    const unsigned int max_epochs = 50000;
    const unsigned int epochs_between_reports = 2;

    if(argc == 2 && !strcmp(argv[1], "gen")) {
        translate_idx_to_fann();
    } else if(argc == 2 && !strcmp(argv[1], "train")) {
        struct fann *ann = fann_create_standard(num_layers, num_input,
            num_neurons_hidden, num_output);

        fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
        fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

        fann_train_on_file(ann, "image.data", max_epochs,
            epochs_between_reports, desired_error);

        fann_save(ann, "image_float.net");

        fann_destroy(ann);
    } else if(argc == 2 && !strcmp(argv[1], "test")) {
        run_tests();
    }
    return 0;
}
