#include <iostream>
#include "timer.h"
#include "icpPointToPlane.h"
#include "icpPointToPoint.h"

using namespace std;

int main(int argc, char **argv)
{
    Timer t1;
    // define a 3 dim problem with 10000 model points
    // and 10000 template points:
    int32_t dim = 2;
    int32_t num = 50;

    // allocate model and template memory
    double *M = (double *)calloc(2 * num, sizeof(double));
    double *T = (double *)calloc(2 * num, sizeof(double));

    int32_t k = 0;
    for (int i = 0; i < num * 2; i += 2)
    {
        M[i] = rand() / (double)RAND_MAX;
        M[i + 1] = sin(M[i]);
        T[i] = M[i] + 0.1;
        T[i + 1] = M[i + 1] - 5;
    }
    t1.start();
    // start with identity as initial transformation
    // in practice you might want to use some kind of prediction here
    Matrix R = Matrix::eye(2);
    Matrix t(2, 1);

    // run point-to-plane ICP (-1 = no outlier threshold)
    // cout << endl
    //      << "Running ICP (point-to-plane, no outliers)" << endl;
    IcpPointToPoint icp(M, num, dim);
    double residual = icp.fit(T, num, R, t, -1);
    t1.stop();
    cout << "Time elapsed: " << t1.getElapsedTimeInMilliSec() << " ms" << endl;
    // results
    cout << endl
         << "Transformation results:" << endl;
    cout << "R:" << endl
         << R << endl
         << endl;
    cout << "t:" << endl
         << t << endl
         << endl;
    cout << "Residual:" << residual;

    // free memory
    free(M);
    free(T);

    // success
    return 0;
}