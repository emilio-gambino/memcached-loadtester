#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

double ADF_Test(double *data, double **coefficients, int N, int D);

double ComputeSE(double **X, double *y, double *coefficients, int n, int d);

void OLS(double **X, const double *y, double *beta, int N, int D);

void InverseMatrix(double **matrix, double **inverse, int n);

// Inputs: a window of data with N samples, and D, the number of coefficients to fit
double ADF_Test(double *data, double **coefficients, int N, int D) {

    // Compute diff values (y_t - y_t-1)
    double diff[N - 1];
    for (int i = 1; i < N; i++) {
        diff[i - 1] = data[i] - data[i - 1];
    } // [D_y1, D_y2, ..., D_y(N-1)]

    int d = D + 2; // Number of lag coefficients (D) + gamma term + intercept + trend
    int n = N - 1 - D; // Number of usable diff samples, i.e. size of y

    // Create regression matrices
    double *y = calloc(n, sizeof(double)); // y is (n x 1)
    double **X = calloc(n, sizeof(double *)); // X is (n x d)
    for (int i = 0; i < n; ++i) {
        X[i] = calloc(d, sizeof(double));
    }

    // Fill up with diff values
    for (int i = 0; i < n; i++) {
        y[i] = diff[i + D];
        X[i][0] = 1.0;  // Intercept \alpha
        X[i][1] = data[i + D + 1];  // Gamma term, what we want to estimate
        for (int j = 2; j < d; j++) {
            X[i][j] = diff[i + D - j + 1];
        }
    }

    // Regression coefficients
    *coefficients = calloc(d, sizeof(double));

    // Fit the coefficients
    OLS(X, y, *coefficients, n, d);

    // Print coefficients
    /*FILE *file = fopen("coeff.txt", "w");
    for (int i = 0; i < d; ++i) {
        printf("coefficients[%d]=%f\n", i, coefficients[i]);
        fprintf(file, "%.6lf\n", coefficients[i]);
    }
    fclose(file);*/

    // Compute Standard error for Gamma
    double SE = ComputeSE(X, y, *coefficients, n, d); // TODO pass index?

    // Compute ADF statistic
    double gamma = (*coefficients)[1];  // Coefficient for lagged difference (y_t-1)
    double ADF = gamma / SE;

    // Free matrices
    for (int i = 0; i < N - D - 1; ++i) {
        free(X[i]);
    }
    free(X);
    free(y);

    return ADF;
}

double *PredictHorizon(double *data, double *coefficients, int d, int horizon) {

    double *predictions = calloc(horizon, sizeof(double));
    double *values = calloc(d - 1 + horizon, sizeof(double)); // Account for intercept coefficient
    memcpy(values, data, d - 1);

    // Predict for the next horizon
    for (int h = 0; h < horizon; ++h) {
        predictions[h] = coefficients[0];
        for (int i = 1; i < d; ++i) {
            predictions[h] += values[h + i - 1] * coefficients[i];
        }
        values[h + d - 1] = predictions[h];
    }

    printf("Values: ");
    for (int i = 0; i < d + horizon - 1; ++i) {
        printf("%f ", values[i]);
    }
    printf("\n");
    free(values);

    return predictions;
}

double ComputeSE(double **X, double *y, double *coefficients, int n, int d) {

    // Compute residuals
    double s2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double res = y[i];
        for (int j = 0; j < d; ++j) {
            res -= coefficients[j] * X[i][j];
        }
        s2 += res * res;
    }
    s2 = s2 / (n - d); // Estimation of simga^2

    // Compute X^TX
    double **XTX = calloc(d, sizeof(double *));
    for (int i = 0; i < d; ++i) {
        XTX[i] = calloc(d, sizeof(double));
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                XTX[j][k] += X[i][j] * X[i][k];
            }
        }
    }
    double **INV = calloc(d, sizeof(double *));
    for (int i = 0; i < d; ++i) {
        INV[i] = calloc(d, sizeof(double));
    }
    InverseMatrix(XTX, INV, d);

    double SE = sqrt(s2 * INV[1][1]);

    for (int i = 0; i < d; ++i) {
        free(XTX[i]);
        free(INV[i]);
    }
    free(XTX);
    free(INV);

    return SE;
}

void OLS(double **X, const double *y, double *beta, int N, int D) {

    // Compute X^TX, X^Ty
    double **XTX = calloc(D, sizeof(double *));
    for (int i = 0; i < D; ++i) {
        XTX[i] = calloc(D, sizeof(double));
    }
    double *XTy = calloc(D, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            for (int k = 0; k < D; k++) {
                XTX[j][k] += X[i][j] * X[i][k];
            }
            XTy[j] += X[i][j] * y[i];
        }
    }

    // Solve X^TX * beta = X^Ty using Gaussian elimination
    for (int i = 0; i < D; i++) {
        for (int j = i + 1; j < D; j++) {
            double factor = XTX[j][i] / XTX[i][i];
            for (int k = 0; k < D; k++) {
                XTX[j][k] -= factor * XTX[i][k];
            }
            XTy[j] -= factor * XTy[i];
        }
    }

    for (int i = D - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < D; j++) {
            sum += XTX[i][j] * beta[j];
        }
        beta[i] = (XTy[i] - sum) / XTX[i][i];
    }

    for (int i = 0; i < D; ++i) {
        free(XTX[i]);
    }
    free(XTX);
    free(XTy);
}

// Function to compute the inverse of a matrix using Gauss-Jordan elimination
void InverseMatrix(double **matrix, double **inverse, int n) {
    // Augment the matrix with the identity matrix
    double **augmented_matrix = (double **) malloc(2 * n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        augmented_matrix[i] = (double *) malloc(2 * n * sizeof(double));
    }

    // Initialize the augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented_matrix[i][j] = matrix[i][j];
            augmented_matrix[i][j + n] = (i == j) ? 1.0 : 0.0; // Identity matrix
        }
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Partial pivoting
        for (int k = i + 1; k < n; k++) {
            if (abs(augmented_matrix[i][i]) < abs(augmented_matrix[k][i])) {
                // Swap rows
                for (int j = 0; j < 2 * n; j++) {
                    double temp = augmented_matrix[i][j];
                    augmented_matrix[i][j] = augmented_matrix[k][j];
                    augmented_matrix[k][j] = temp;
                }
            }
        }

        // Make the diagonal elements of the current column equal to 1
        double divisor = augmented_matrix[i][i];
        for (int j = 0; j < 2 * n; j++) {
            augmented_matrix[i][j] /= divisor;
        }

        // Make other elements of the current column equal to 0
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double multiplier = augmented_matrix[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented_matrix[k][j] -= multiplier * augmented_matrix[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = augmented_matrix[i][j + n];
        }
    }

    // Free memory
    for (int i = 0; i < n; i++) {
        free(augmented_matrix[i]);
    }
    free(augmented_matrix);
}

/*int main() {

    double pred;
    int method;
    FILE *fptr;

    if ((fptr = fopen("testdata.txt", "r")) == NULL) {
        fprintf(stderr, "Unable to open data file\n");
        exit(0);
    }

    int N = 1000;
    int stop = 200;
    int D = 6;
    double *data = calloc(N, sizeof(double));

    int i = 0;
    while (fscanf(fptr, "%lf", &data[i]) != EOF) {
        i++;
        if (i > stop) {
            break;
        }
    }
    printf("Read %d points\n", i);
    fclose(fptr);

    double adf = ADF_Test(data, stop, D);
    printf("ADF stat: %lf\n", adf);

    // Sample data
    /*FILE *fileX = fopen("X_coeff.txt", "r");
    FILE *fileY = fopen("y_coeff.txt", "r");

    double **X = (double **) malloc(7 * sizeof(double *));
    //double *y = (double *) malloc(20 * sizeof(double));
    // Read X coefficients
    for (int i = 0; i < 7; i++) {
        X[i] = (double *) malloc(7 * sizeof(double));
        for (int j = 0; j < 7; j++) {
            if (fscanf(fileX, "%lf", &X[i][j]) != 1) {
                printf("Error reading from file\n");
                return 1;
            }
            printf("%lf ", X[i][j]);
        }
        printf("\n");
    }
    // Read y coefficients
    for (int i = 0; i < 20; i++) {
        if (fscanf(fileY, "%lf", &y[i]) != 1) {
            printf("Error reading from file\n");
            return 1;
        }
    }
    // Close files
    fclose(fileX);
    fclose(fileY);

    double **I = calloc(7, sizeof(double *));
    for (int i = 0; i < 7; ++i) {
        I[i] = calloc(7, sizeof(double));
    }
    printf("Trying to inverse\n");
    inverse_matrix(X, I, 7);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            printf("%lf ", I[i][j]);
        }
        printf("\n");
    }

    double **J = calloc(7, sizeof(double *));
    for (int i = 0; i < 7; ++i) {
        J[i] = calloc(7, sizeof(double));
    }


    inverse_matrix(I, J, 7);


    printf("Should give Identity: \n");
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            double r = 0.0;
            for (int k = 0; k < 7; k++) {
                r += I[i][k] * J[k][j];
            }
            printf("%lf ", r);
        }
        printf("\n");
    }

    printf("Inverse of Inverse:\n");
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            printf("%lf ", J[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 7; ++i) {
        free(J[i]);
    }
    free(J);

    for (int i = 0; i < 7; ++i) {
        free(I[i]);
    }
    free(I);


    // TODO implement inside of data-caching
    // TODO experiment with different degrees

    // TODO optimize OLS and matrix inverse so that don't need to recompute

    free(data);

    return 0;
}*/
