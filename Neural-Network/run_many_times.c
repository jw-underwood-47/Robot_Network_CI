#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -a, --accuracy <float>          (default: 1.0)\n");
    printf("  -b, --batch_size <v1> [v2]      (default: 32 32)\n");
    printf("  -e, --epochs <v1> [v2]          (default: 800 800)\n");
    printf("  -g, --print_gap <int>           (default: 2)\n");
    printf("  -l, --learning_rate <v1> [v2]   (default: 0.02 0.02)\n");
    printf("  -p, --test_accuracy <float>     (default: 1.0)\n");
    printf("  -r, --run_num <int>             (default: 0)\n");
    printf("  -s, --stratified                (flag)\n");
    printf("  -t, --test_num <int>            (default: 0)\n");
    printf(" Additional info: epochs increment by 10.\n Batch size doubles each time.\n Learning rate increments by 0.01\n");
}

int main(int argc, char *argv[]) {
    int opt, bs = 0, epoch = 0, second_epoch = 0, second_bs = 0;
    float def_lr = 0.02, lr = 0, second_lr = 0;
    int def_bs = 32, def_epoch = 800;
    /* Defaults */
    char *accuracy = "1.0";

    char *batch_size[2] = {"32", "32"};
    char *epochs[2] = {"800", "800"};
    char *learning_rate[2] = {"0.02", "0.02"};

    char *print_gap = "2";
    char *test_accuracy = "1.0";
    char *run_num = "0";
    char *test_num = "0";
    int stratified = 0;

    static struct option long_options[] = {
        {"accuracy",      required_argument, 0, 'a'},
        {"batch_size",    required_argument, 0, 'b'},
        {"epochs",        required_argument, 0, 'e'},
        {"print_gap",     required_argument, 0, 'g'},
        {"learning_rate", required_argument, 0, 'l'},
        {"test_accuracy", required_argument, 0, 'p'},
        {"run_num",       required_argument, 0, 'r'},
        {"stratified",    no_argument,       0, 's'},
        {"test_num",      required_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "a:b:e:g:l:p:r:st:",
        long_options, NULL)) != -1) {

        switch (opt) {

            case 'a':
                accuracy = optarg;
                break;

            case 'b':
                batch_size[0] = optarg;
                bs = atoi(batch_size[0]);
                if (optind < argc && argv[optind][0] != '-') {
                    batch_size[1] = argv[optind++];
                    second_bs = atoi(batch_size[1]);
                }
                break;

            case 'e':
                epochs[0] = optarg;
                epoch = atoi(epochs[0]);
                if (optind < argc && argv[optind][0] != '-') {
                    epochs[1] = argv[optind++];
                    second_epoch = atoi(epochs[1]);
                }
                break;

            case 'l':
                learning_rate[0] = optarg;
                lr = atof(learning_rate[0]);
                if (optind < argc && argv[optind][0] != '-') {
                    learning_rate[1] = argv[optind++];
                    second_lr = atof(learning_rate[1]);
                }
                break;

            case 'g':
                print_gap = optarg;
                break;

            case 'p':
                test_accuracy = optarg;
                break;

            case 'r':
                run_num = optarg;
                break;

            case 's':
                stratified = 1;
                break;

            case 't':
                test_num = optarg;
                break;

            default:
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        }
        }

        /* Example command construction */
        char command[512];

        printf("\nParsed values:\n");
        printf("accuracy = %s\n", accuracy);
        printf("batch_size = %s %s\n", batch_size[0], batch_size[1]);
        printf("epochs = %s %s\n", epochs[0], epochs[1]);
        printf("learning_rate = %s %s\n",
               learning_rate[0], learning_rate[1]);
        printf("print_gap = %s\n", print_gap);
        printf("test_accuracy = %s\n", test_accuracy);
        printf("run_num = %s\n", run_num);
        printf("test_num = %s\n", test_num);
        printf("stratified = %s\n", stratified ? "true" : "false");

        if (epoch == 0) epoch = def_epoch;
        if (second_epoch == 0) second_epoch = epoch;
        if (bs == 0) bs = def_bs;
        if (second_bs == 0) second_bs = bs;
        if (lr == 0) lr = def_lr;
        if (second_lr == 0) second_lr = lr;
        printf("set new values: %d %d %f %d %d %f\n", bs, epoch, lr, second_bs, second_epoch, second_lr);

        for (int i = epoch; i <= second_epoch; i+=10){
            //printf("in first for loop\n");
            for (int j = bs; j <= second_bs; j*=2){
                //printf("in second for loop\n");
                for (float k = lr; k <= second_lr; k+=0.01){
                    sprintf(command, "python3 main.py -a %s -b %d -e %d -l %f -g %s -p %s -r %s -t %s %s", accuracy, j, i, k, print_gap, test_accuracy, run_num, test_num, stratified ? "-s" : "");
                    printf("Calling command\n%s\n", command);
                    system(command);
                }
            }
        }

        return 0;
}
