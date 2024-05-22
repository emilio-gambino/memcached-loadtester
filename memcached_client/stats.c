//
//  stats.c
//
//  Author: David Meisner (meisner@umich.edu)
//

#define MAX_ITER 5000

#include "stats.h"
#include "loader.h"
#include <assert.h>
#include "worker.h"
#include "PID.h"
#include "predict.h"

pthread_mutex_t stats_lock = PTHREAD_MUTEX_INITIALIZER;
struct timeval start_time;
struct memcached_stats global_stats;
PIDController pid;

double latencies[MAX_ITER] = {0.0};
int curr_iter = 0;

void addSample(struct stat *stat, float value) {
    stat->s0 += 1.0;
    stat->s1 += value;
    stat->s2 += value * value;
    stat->min = fmin(stat->min, value);
    stat->max = fmax(stat->max, value);

    if (value < .001) {
        int bin = (int) (value * 10000000);
        stat->micros[bin] += 1;
    } else if (value < 5.0) {
        int bin = value * 10000.0;
        assert(bin < 50001);
        stat->millis[bin] += 1;
    } else if (value < 999) {
        int bin = (int) value;
        stat->fulls[bin] += 1;
    } else {
        int bin = (int) value / 1000;
        if (bin > 999) {
            bin = 999;
        }
        stat->fulls[bin] += 1;
    }
}//End addAvgSample()

double getAvg(struct stat *stat) {
    return (stat->s1 / stat->s0);
}//End getAvg()

double getStdDev(struct stat *stat) {
    return sqrt((stat->s0 * stat->s2 - stat->s1 * stat->s1) / (stat->s0 * (stat->s0 - 1)));
}//End getStdDev()

//Should we exit because time has expired?
void checkExit(struct config *config) {

    int runTime = config->run_time;
    struct timeval currentTime;
    gettimeofday(&currentTime, NULL);
    double totalTime = currentTime.tv_sec - start_time.tv_sec + 1e-6 * (currentTime.tv_sec - start_time.tv_sec);
    if (totalTime >= runTime && runTime > 0) {
        printf("Ran for %f, exiting\n", totalTime);
        exit(0);
    }

}//End checkExit()

double findQuantile(struct stat *stat, double quantile) {

    //Find the 95th-percentile
    int nTillQuantile = global_stats.response_time.s0 * quantile;
    int count = 0;
    int i;
    for (i = 0; i < 10000; i++) {
        count += stat->micros[i];
        if (count >= nTillQuantile) {
            double quantile = (i + 1) * .0000001;
            return quantile;
        }
    }//End for i

    for (i = 0; i < 50000; i++) {
        count += stat->millis[i];
        if (count >= nTillQuantile) {
            double quantile = (i + 1) * .0001;
            return quantile;
        }
    }//End for i
    printf("count  %d\n", count);

    for (i = 0; i < 1000; i++) {
        count += stat->fulls[i];
        if (count >= nTillQuantile) {
            double quantile = i + 1;
            return quantile;
        }
    }//End for i
    return 1000;

}//End findQuantile()

void printGlobalStats(struct config *config) {

    pthread_mutex_lock(&stats_lock);
    struct timeval currentTime;
    gettimeofday(&currentTime, NULL);
    double timeDiff = currentTime.tv_sec - global_stats.last_time.tv_sec +
                      1e-6 * (currentTime.tv_sec - global_stats.last_time.tv_sec);
    double rps = global_stats.requests / timeDiff;
    double std = getStdDev(&global_stats.response_time);
    double q90 = findQuantile(&global_stats.response_time, .90);
    double q95 = findQuantile(&global_stats.response_time, .95);
    double q99 = findQuantile(&global_stats.response_time, .99);
    double average = 1000 * getAvg(&global_stats.response_time);

    printf("%10s,%10s,%8s,%16s, %8s,%11s,%10s,%13s,%10s,%10s,%10s,%12s,%10s,%10s,%11s,%14s\n", "unix_ts", "timeDiff",
           "rps", "requests", "gets", "sets", "hits", "misses", "avg_lat", "90th", "95th", "99th", "std", "min", "max",
           "avgGetSize");
    printf("%10ld, %10f, %9.1f,  %10d, %10d, %10d, %10d, %10d, %10f, %10f, %10f, %10f, %10f, %10f, %10f, %10f\n",
           currentTime.tv_sec, timeDiff, rps, global_stats.requests, global_stats.gets, global_stats.sets,
           global_stats.hits, global_stats.misses, average, 1000 * q90, 1000 * q95, 1000 * q99, 1000 * std,
           1000 * global_stats.response_time.min, 1000 * global_stats.response_time.max,
           getAvg(&global_stats.get_size));
    int i;
    printf("Outstanding requests per worker:\n");
    for (i = 0; i < config->n_workers; i++) {
        printf("%d ", config->workers[i]->n_requests);
    }
    printf("\n");

    double measurement;
    if (config->measurement == 95)
        measurement = 1000 * q95;
    else if (config->measurement == 99)
        measurement = 1000 * q99;
    else
        measurement = 1000 * q90;

    latencies[curr_iter] = measurement;

    int horizon = 5;
    int num_samples = 50; // Number of regression samples
    if (curr_iter >= num_samples && curr_iter % horizon == 0) {
        printf("Enough samples, starting regression!\n");
        // 1. Get AR coefficients
        double *coefficients;
        double *regression_data = (latencies + curr_iter - num_samples);
        double *prediction_data = (latencies + curr_iter - config->degree);
        double adf = ADF_Test(latencies + (curr_iter - Nn), &coefficients, num_samples, config->degree);
        // 2. Check for ADF statistic
        printf("ADF stat: %f\n", adf);
        // 3. If stationary, predict for next horizon
        double *predicted = PredictHorizon(prediction_data, coefficients, config->degree, horizon);

        // Save the predictions
        // 4. Adapt input using controller
        printf("Predictions: ");
        for (int i = 0; i < horizon; ++i) {
            printf("%f ", predicted[i]);
        }
        printf("\n");

        free(coefficients);
        free(predicted);
    }

    if (config->SLO != -1) {
        // Update PID struct
        PIDController_Update(&pid, config->SLO, measurement);
        printf("Setting new RPS to: %f\n", pid.out);

        // Update worker distribution
        int meanInterarrival = 1.0 / (((float) pid.out) / (float) config->n_workers) * 1e6;
        for (i = 0; i < config->n_workers; i++) {
            config->workers[i]->config->interarrival_dist = createExponentialDistribution(meanInterarrival);
        }
    }

    if (curr_iter >= MAX_ITER) {
        printf("MAX ITER REACHED!\n");
        assert(0);
    }

    //Reset stats
    memset(&global_stats, 0, sizeof(struct memcached_stats));
    global_stats.response_time.min = 1000000;
    global_stats.last_time = currentTime;

    checkExit(config);
    pthread_mutex_unlock(&stats_lock);

}//End printGlobalStats()


//Print out statistics every second
void statsLoop(struct config *config) {

    pthread_mutex_lock(&stats_lock);
    gettimeofday(&start_time, NULL);
    pthread_mutex_unlock(&stats_lock);

    // TODO tune controller parameters
    /* Controller parameters */
    float PID_KP = 50000.0f;
    float PID_KI = 2000.0f;
    float PID_KD = 200.0f;
    float PID_TAU = 0.02f;
    pid = (PIDController) {PID_KP, PID_KI, PID_KD, PID_TAU, 50000, 800000, 0, 100000, config->stats_time};

    printf("Creating PID controller with KP=%f, KI=%f, KD=%f\n", PID_KP, PID_KI, PID_KD);

    sleep(2);
    printf("Stats:\n");
    printf("-------------------------\n");
    while (1) {
        if (curr_iter % config->window == 0) {
            printf("Iteration: %d\n", curr_iter);
            printGlobalStats(config);
        }
        sleep(config->stats_time);
        ++curr_iter;
    }//End while()

    // Problems with PID
    // 1. Unstable -> Take larger windows
    // 2. Non linear, especially at kneecap point -> oscillation not well compensated by integral term

    // TODO estimate the variance based on each iterations
    // TODO figure out what window size to aggregate over for accurate estimation
    // The greater the variance, the longer the window size should be

    // Kneecap detection??

    // TODO add ADF test code + run ADF test every window size > degree

    // TODO Add code to both see immediate sample and aggregate over window

    // TODO change to compute SLO over window requests
    // TODO every time curr_iter % window_size == 0, run ADF code on window
    // TODO if ADF is below a certain threshold, compute tail latency aggregate

    // TODO set better bounds min max
    // TODO as we get closer to SLO, or more variance, take in more samples
    // TODO compute/print single iteration latency aswell as aggregate one

    // TODO aggregation across windows
    // TODO feed aggregated tail latency inside PID controller

    // TODO add ADF test code + run ADF test every window size > degree
    // 1. Run the workload for window size, save latency at each iteration
    // 2. Compute ADF for that window
    // 3. Printf ADF stat, and potential critical value
}//End statisticsLoop()

