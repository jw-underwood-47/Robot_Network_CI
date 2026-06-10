#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/wait.h>
#include <errno.h>

#define DEF_EPOCH_INCREMENT 10
#define MAX_JOBS_DEFAULT    1

void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -a, --accuracy <float>          (default: 1.0)\n");
    printf("  -b, --batch_size <v1> [v2]      (default: 32 32)\n");
    printf("  -e, --epochs <v1> [v2] [incr]   (default: 800 800 10)\n");
    printf("  -g, --print_gap <int>            (default: 2)\n");
    printf("  -j, --jobs <int>                 max parallel processes (default: 1)\n");
    printf("  -l, --learning_rate <v1> [v2]   (default: 0.02 0.02)\n");
    printf("  -p, --test_accuracy <float>     (default: 1.0)\n");
    printf("  -r, --run_num <int>              (default: 0)\n");
    printf("  -s, --stratified                 (flag)\n");
    printf("  -t, --test_num <int>             (default: 0)\n");
    printf(" Additional info: epochs increment by 10 unless modified with the third number ([incr]) after -e.\n"
    " Batch size doubles each time.\n"
    " Learning rate increments by 0.01\n");
}

/*
 * Spawn one child running:
 *   python3 main.py -a <accuracy> -b <bs> -e <epoch> -l <lr>
 *                   -g <print_gap> -p <test_accuracy>
 *                   -r <run_num> -t <test_num> [-s]
 *
 * All arguments are passed as discrete execvp() tokens — no shell
 * is involved, so there is no command-injection surface.
 *
 * Returns the child PID on success, -1 on fork failure.
 */
static pid_t spawn_child(const char *accuracy, int bs, int epoch, float lr,
                         const char *print_gap, const char *test_accuracy,
                         const char *run_num, const char *test_num,
                         int stratified)
{
    /* Pre-format the numeric arguments into fixed buffers. */
    char bs_str[32], epoch_str[32], lr_str[32];
    snprintf(bs_str,    sizeof(bs_str),    "%d", bs);
    snprintf(epoch_str, sizeof(epoch_str), "%d", epoch);
    snprintf(lr_str,    sizeof(lr_str),    "%f", lr);

    /*
     * Build the argv array.  The stratified flag occupies the last
     * optional slot; we size for the worst case.
     */
    const char *args[32];
    int idx = 0;
    args[idx++] = "python3";
    args[idx++] = "main.py";
    args[idx++] = "-a"; args[idx++] = accuracy;
    args[idx++] = "-b"; args[idx++] = bs_str;
    args[idx++] = "-e"; args[idx++] = epoch_str;
    args[idx++] = "-l"; args[idx++] = lr_str;
    args[idx++] = "-g"; args[idx++] = print_gap;
    args[idx++] = "-p"; args[idx++] = test_accuracy;
    args[idx++] = "-r"; args[idx++] = run_num;
    args[idx++] = "-t"; args[idx++] = test_num;
    if (stratified)
        args[idx++] = "-s";
    args[idx] = NULL;

    /* Log the command we are about to run. */
    printf("Spawning:");
    for (int i = 0; args[i] != NULL; i++)
        printf(" %s", args[i]);
    printf("\n");
    fflush(stdout);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }
    if (pid == 0) {
        /* Child: replace image with python3. */
        execvp("python3", (char * const *)args);
        /* execvp only returns on error. */
        perror("execvp");
        _exit(EXIT_FAILURE);
    }
    /* Parent: return child PID so the caller can track it. */
    return pid;
}

/*
 * Wait for at least one child to finish, decrementing *running.
 * Called whenever we have hit the jobs limit before spawning another.
 */
static void wait_for_one(int *running)
{
    int status;
    pid_t finished = wait(&status);
    if (finished < 0) {
        if (errno != ECHILD)
            perror("wait");
    } else {
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
            fprintf(stderr, "Child %d exited with status %d\n",
                    (int)finished, WEXITSTATUS(status));
            else if (WIFSIGNALED(status))
                fprintf(stderr, "Child %d killed by signal %d\n",
                        (int)finished, WTERMSIG(status));
                (*running)--;
    }
}

int main(int argc, char *argv[]) {
    int opt;
    int bs = 0, epoch = 0, epoch_increment = 0, max_epoch = 0, second_bs = 0;
    float def_lr = 0.02f, lr = 0.0f, second_lr = 0.0f;
    int def_bs = 32, def_epoch = 800;
    int max_jobs = MAX_JOBS_DEFAULT;

    /* Defaults (stored as strings for direct pass-through to child argv). */
    const char *accuracy      = "1.0";
    const char *print_gap     = "2";
    const char *test_accuracy = "1.0";
    const char *run_num       = "0";
    const char *test_num      = "0";
    int stratified = 0;

    char *batch_size[2]    = {"32",   "32"};
    char *epochs[3]        = {"800",  "800", "10"};
    char *learning_rate[2] = {"0.02", "0.02"};

    static struct option long_options[] = {
        {"accuracy",      required_argument, 0, 'a'},
        {"batch_size",    required_argument, 0, 'b'},
        {"epochs",        required_argument, 0, 'e'},
        {"print_gap",     required_argument, 0, 'g'},
        {"jobs",          required_argument, 0, 'j'},
        {"learning_rate", required_argument, 0, 'l'},
        {"test_accuracy", required_argument, 0, 'p'},
        {"run_num",       required_argument, 0, 'r'},
        {"stratified",    no_argument,       0, 's'},
        {"test_num",      required_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "a:b:e:g:j:l:p:r:st:",
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
                    max_epoch = atoi(epochs[1]);
                    if (optind < argc && argv[optind][0] != '-') {
                        epochs[2] = argv[optind++];
                        epoch_increment = atoi(epochs[2]);
                    }
                }
                break;

            case 'g':
                print_gap = optarg;
                break;

            case 'j':
                max_jobs = atoi(optarg);
                if (max_jobs < 1) {
                    fprintf(stderr, "Error: --jobs must be >= 1\n");
                    exit(EXIT_FAILURE);
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

        /* Apply defaults for any values not set by the user. */
        if (epoch == 0)           epoch           = def_epoch;
        if (max_epoch == 0)       max_epoch       = epoch;
        if (epoch_increment == 0) epoch_increment = DEF_EPOCH_INCREMENT;
        if (bs == 0)              bs              = def_bs;
        if (second_bs == 0)       second_bs       = bs;
        if (lr == 0.0f)           lr              = def_lr;
        if (second_lr == 0.0f)    second_lr       = lr;

        printf("\nParsed values:\n");
    printf("  accuracy      = %s\n", accuracy);
    printf("  batch_size    = %s %s\n", batch_size[0], batch_size[1]);
    printf("  epochs        = %s %s %s\n", epochs[0], epochs[1], epochs[2]);
    printf("  learning_rate = %s %s\n", learning_rate[0], learning_rate[1]);
    printf("  print_gap     = %s\n", print_gap);
    printf("  test_accuracy = %s\n", test_accuracy);
    printf("  run_num       = %s\n", run_num);
    printf("  test_num      = %s\n", test_num);
    printf("  stratified    = %s\n", stratified ? "true" : "false");
    printf("  max_jobs      = %d\n", max_jobs);
    printf("  effective: bs=%d..%d  epochs=%d..%d (step %d)  lr=%.4f..%.4f\n\n",
           bs, second_bs, epoch, max_epoch, epoch_increment, lr, second_lr);

    int running = 0;  /* Number of currently live child processes. */

    for (int i = epoch; i <= max_epoch; i += epoch_increment) {
        for (int j = bs; j <= second_bs; j *= 2) {
            for (float k = lr; k <= second_lr + 1e-9f; k += 0.01f) {
                /* Throttle: wait for a slot before spawning. */
                while (running >= max_jobs)
                    wait_for_one(&running);

                pid_t pid = spawn_child(accuracy, j, i, k,
                                        print_gap, test_accuracy,
                                        run_num, test_num, stratified);
                if (pid > 0)
                    running++;
            }
        }
    }

    /* Wait for all remaining children. */
    while (running > 0)
        wait_for_one(&running);

    return 0;
}
