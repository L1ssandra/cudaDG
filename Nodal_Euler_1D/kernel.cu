#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include<graphics.h>
#include<conio.h>
#include<stdlib.h>
#include<time.h>
#include <chrono>


//***************************************************************************************
/* 

The 1D nodal discontinuous Galerkin method solving Euler equation:

 rho_t +             m_x = 0
   m_t + (rho*u^2 + p)_x = 0
   E_t +    (u(E + p))_x = 0

 with the ideal gas law

 E = p/(gamma - 1) + 0.5*rho*u^2, where gamma is the gas constant (usually taken as 1.4).

*/
//***************************************************************************************



//***************************************************************************************
/* Parameters */
//***************************************************************************************

int const Nx = 1024;
int const k = 3;
int const NumEq = 3;
double const CFL = 5.0 / (2 * k + 1);
int const RKorder = 4;
int const NumGLP = k + 1;
int const plot = 1;
double const gamma = 1.4;
double const gamma1 = gamma - 1;

// The limiter
double const c0 = 0.01;

// CUDA data !
int const NB = 32; // Number of block

// draw the picture
int const frame = 400;
double const scalex = 100;
double const scaley = 60;


int const dimPk = k + 1;
int const Nx1 = Nx + 1;
double const pi = 4 * atan(1.0);
int const NT = Nx / NB; // Number of threads in each block


/*
Examples:

1: (1 + 0.2*sin(x),1,1) in [0,2*pi] with periodic boundaries.
This example is always used for accuracy tests.

2: Sod shock tube. The computational domain is [-1,1] with outflow (or Dirichlet) boundaries.
The initial data is given by
rhoL = 1, rhoR = 0.125
  uL = 0,   uR = 0
  pL = 1,   uR = 0.1
This example contains a minor shock. Entropy stable schemes can hand this shock.

*/
int const example = 3;


// The numerical solution
double hx, hx1;
double Xc[Nx1 + 1];
double t, t1, dt, alpha;
double L1_Error = 0, L2_Error = 0, L8_Error = 0, uE;
double uumax, uumin;


// The basis
double lambda[NumGLP], weight[NumGLP];
double Dmat[NumGLP][NumGLP];


// The loop variables
int i, j, i1, d, ii1, n;


//***************************************************************************************
/* The initial data */
//***************************************************************************************
class init
{
public:

    double const pi = 4 * atan(1.0);
    double xa, xb, tend;
    int bcL, bcR;

    void set_init()
    {

        if (example == 1)
        {
            xa = 0;
            xb = 2 * pi;
            bcL = 1;
            bcR = 1;
            tend = 2 * pi;
        }
        else if (example == 2)
        {
            xa = -1;
            xb = 1;
            bcL = 100;
            bcR = 100;
            tend = 0.4;
        }
        else if (example == 3)
        {
            xa = -5;
            xb = 5;
            bcL = 100;
            bcR = 100;
            tend = 1.8;
        }

    }

    double rho0(double x)
    {
        if (example == 1)
        {
            return 1 + 0.2*sin(x);
        }
        else if (example == 2)
        {
            if (x < 0)
            {
                return 1;
            }
            else
            {
                return 0.125;
            }
        }
        else if (example == 3)
        {
            if (x < -4)
            {
                return 3.857143;
            }
            else
            {
                return 1 + 0.2 * sin(5 * x);
            }
        }
    }

    double u0(double x)
    {
        if (example == 1)
        {
            return 1;
        }
        else if (example == 2)
        {
            return 0;
        }
        else if (example == 3)
        {
            if (x < -4)
            {
                return 2.629369;
            }
            else
            {
                return 0;
            }
        }
    }

    double p0(double x)
    {
        if (example == 1)
        {
            return 1;
        }
        else if (example == 2)
        {
            if (x < 0)
            {
                return 1;
            }
            else
            {
                return 0.1;
            }
        }
        else if (example == 3)
        {
            if (x < -4)
            {
                return 10.3333;
            }
            else
            {
                return 1;
            }
        }

    }

    double U1(double x)
    {
        return rho0(x);
    }

    double U2(double x)
    {
        return rho0(x) * u0(x);
    }

    double U3(double x)
    {
        return p0(x) / gamma1 + 0.5 * rho0(x) * u0(x) * u0(x);
    }

};
init data;

//***************************************************************************************
/* The flux function */
//***************************************************************************************

__host__ __device__ double f1(double rho, double m, double E, double gamma1)
{
    return m;
}

__host__ __device__ double f2(double rho, double m, double E, double gamma1)
{
    double u, p;

    u = m / rho;
    p = gamma1 * (E - 0.5 * rho * u * u);

    return rho * u * u + p;
}

__host__ __device__ double f3(double rho, double m, double E, double gamma1)
{
    double u, p;

    u = m / rho;
    p = gamma1 * (E - 0.5 * rho * u * u);

    return u * (E + p);
}


//***************************************************************************************
/* The wave speeds */
//***************************************************************************************

__host__ __device__ double maxspeed(double rho, double m, double E, double gamma, double gamma1)
{

    double u, p, c;

    u = m / rho;
    p = gamma1 * (E - 0.5 * rho * u * u);

    c = pow(abs(gamma * p / rho), 0.5);

    return u + c;
}


__host__ __device__ double minspeed(double rho, double m, double E, double gamma, double gamma1)
{

    double u, p, c;

    u = m / rho;
    p = gamma1 * (E - 0.5 * rho * u * u);

    c = pow(abs(gamma * p / rho), 0.5);

    return u - c;
}


__host__ __device__ double maxabsspeed(double rho, double m, double E, double gamma, double gamma1)
{

    double u, p, c;

    u = m / rho;
    p = gamma1 * (E - 0.5 * rho * u * u);

    c = pow(abs(gamma * p / rho), 0.5);

    return abs(u) + c;
}



void set_bc(double uh[Nx1 + 1][NumGLP][NumEq])
{

    // Set boundary condition !
    if (data.bcL == 1)
    {
        for (d = 0; d < NumGLP; d++)
        {
            for (n = 0; n < NumEq; n++)
            {
                uh[0][d][n] = uh[Nx][d][n];
            }
        }
    }
    else if (data.bcL == 2)
    {
        for (d = 0; d < NumGLP; d++)
        {
            for (n = 0; n < NumEq; n++)
            {
                uh[0][d][n] = uh[1][0][n];
            }
        }
    }
    

    if (data.bcR == 1)
    {
        for (d = 0; d < NumGLP; d++)
        {
            for (n = 0; n < NumEq; n++)
            {
                uh[Nx1][d][n] = uh[1][d][n];
            }
        }
    }
    else if (data.bcR == 2)
    {
        for (d = 0; d < NumGLP; d++)
        {
            for (n = 0; n < NumEq; n++)
            {
                uh[Nx1][d][n] = uh[Nx][0][n];
            }
        }
    }

}

//***************************************************************************************
/* The jump filter */
//***************************************************************************************
void Limiter(double uh[Nx1 + 1][NumGLP][NumEq], double dt)
{
    double theta, theta0, theta1, theta2, theta3[NumEq];
    double udxjump[Nx1][NumGLP][NumEq];
    double uhdxR[Nx1 + 1][NumGLP][NumEq], uhdxL[Nx1 + 1][NumGLP][NumEq];
    double Duh[Nx1 + 1][NumGLP][NumGLP][NumEq];
    double uhbar[Nx1 + 1][NumEq];
    double enth[Nx1 + 1],p;
    int L;

    // calculate the derivatives
    for (i = 0; i <= Nx1; i++)
    {
        for (n = 0; n < NumEq; n++)
        {
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                Duh[i][i1][0][n] = uh[i][i1][n];
            }
            for (d = 1; d < NumGLP; d++)
            {
                for (i1 = 0; i1 < NumGLP; i1++)
                {
                    Duh[i][i1][d][n] = 0;
                    for (L = 0; L < NumGLP; L++)
                    {
                        Duh[i][i1][d][n] = Duh[i][i1][d][n] + Dmat[i1][L] * Duh[i][L][d - 1][n];
                    }
                }
            }
        }
        
    }

    // calcualte the value at interfaces
    for (i = 0; i <= Nx; i++)
    {
        for (d = 0; d < NumGLP; d++)
        {
            for (n = 0; n < NumEq; n++)
            {
                uhdxR[i][d][n] = Duh[i][k][d][n];
                uhdxL[i + 1][d][n] = Duh[i + 1][0][d][n];
            }
        }
    }

    // calculate the jump
    for (i = 0; i <= Nx; i++)
    {
        for (d = 0; d < NumGLP; d++)
        {
            for (n = 0; n < NumGLP; n++)
            {
                udxjump[i][d][n] = abs(uhdxL[i + 1][d][n] - uhdxR[i][d][n]);
            }
        }
    }

    // calculate uhbar
    for (i = 1; i <= Nx; i++)
    {
        for (n = 0; n < NumEq; n++)
        {
            uhbar[i][n] = 0;
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                uhbar[i][n] = uhbar[i][n] + 0.5 * weight[i1] * uh[i][i1][n];
            }
        }
        p = gamma1 * (uhbar[i][2] - 0.5 * uhbar[i][1] * uhbar[i][1] / uhbar[i][0]);
        enth[i] = (uhbar[i][2] + p) / uhbar[i][0];
    }


    for (i = 1; i <= Nx; i++)
    {
        for (n = 0; n < NumEq; n++)
        {
            //theta0 = c0 * abs(HCL.df(uh[i][0])) * ((ujump[i - 1] + ujump[i])) / hx;
            //theta1 = c0 * abs(HCL.df(uh[i][0])) * ((ujump[i - 1] + ujump[i]) + 2 * (uxjump[i - 1] + uxjump[i])) / hx;
            //theta2 = c0 * abs(HCL.df(uh[i][0])) * ((ujump[i - 1] + ujump[i]) + 2 * (uxjump[i - 1] + uxjump[i]) + 6 * (uxxjump[i - 1] + uxxjump[i])) / hx;
            theta3[n] = 0;
            for (d = 0; d < NumGLP; d++)
            {
                theta3[n] = theta3[n] + c0 / enth[i] * fmax(d, 1) * (d + 1) * (udxjump[i - 1][d][n] + udxjump[i][d][n]) / hx;
                //printf("%.16f  %.16f\n", udxjump[i - 1][d], udxjump[i][d]);
            }
        }

        theta = 0;
        for (n = 0; n < NumEq; n++)
        {
            theta = fmax(theta, theta3[n]);
        }

        for (n = 0; n < NumEq; n++)
        {
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                uh[i][i1][n] = uhbar[i][n] + exp(-theta * dt) * (uh[i][i1][n] - uhbar[i][n]);
            }
        }
    }

}



//***************************************************************************************
/* Discontinuous Galerkin space discretization */
//***************************************************************************************

__global__ void Lh(double uh[Nx1 + 1][NumGLP][NumEq], double du[Nx1 + 1][NumGLP][NumEq], double hx, double weight[NumGLP], double Dmat[NumGLP][NumGLP], double gamma, double gamma1)
{
    double fL[NumEq], fR[NumEq], fLp[NumEq], fRn[NumEq], SR, SL;
    double fhatR[NumEq], fhatL[NumEq];
    double uhR[NumEq], uhRn[NumEq], uhLp[NumEq], uhL[NumEq];
    double fint[NumGLP][NumEq];
    int i, i1, d, L, n;

    i = blockIdx.x * blockDim.x + threadIdx.x + 1; // i from 1 to Nx

    // Compute F(U) at each nodal point
    for (i1 = 0; i1 < NumGLP; i1++)
    {
        fint[i1][0] = f1(uh[i][i1][0], uh[i][i1][1], uh[i][i1][2], gamma1);
        fint[i1][1] = f2(uh[i][i1][0], uh[i][i1][1], uh[i][i1][2], gamma1);
        fint[i1][2] = f3(uh[i][i1][0], uh[i][i1][1], uh[i][i1][2], gamma1);
    }

    // Calculate the volume integral: \int_Ii f(uh)*phi' dx !
    for (i1 = 0; i1 < NumGLP; i1++)
    {
        for (n = 0; n < NumEq; n++)
        {
            du[i][i1][n] = 0;
            for (L = 0; L < NumGLP; L++)
            {
                du[i][i1][n] = du[i][i1][n] - (2.0 / hx) * Dmat[i1][L] * fint[L][n];
            }
        }
    }

    // The surface values: uhR[1:Nx], uhL[1:Nx] !
    for (n = 0; n < NumEq; n++)
    {
        uhR[n] = uh[i][k][n];
        uhL[n] = uh[i][0][n];
        uhLp[n] = uh[i + 1][0][n];
        uhRn[n] = uh[i - 1][k][n];
    }

    // Calculate numerical flux: fhat[0:Nx] !
    
    fR[0] = f1(uhR[0], uhR[1], uhR[2], gamma1);
    fR[1] = f2(uhR[0], uhR[1], uhR[2], gamma1);
    fR[2] = f3(uhR[0], uhR[1], uhR[2], gamma1);

    fLp[0] = f1(uhLp[0], uhLp[1], uhLp[2], gamma1);
    fLp[1] = f2(uhLp[0], uhLp[1], uhLp[2], gamma1);
    fLp[2] = f3(uhLp[0], uhLp[1], uhLp[2], gamma1);

    SR = fmax(abs(maxabsspeed(uhR[0], uhR[1], uhR[2], gamma, gamma1)), abs(maxabsspeed(uhLp[0], uhLp[1], uhLp[2], gamma, gamma1)));

    // The Lax-Friedrichs flux !
    for (n = 0; n < NumEq; n++)
    {
        fhatR[n] = 0.5 * (fLp[n] + fR[n]) - 0.5 * SR * (uhLp[n] - uhR[n]);
    }

    fL[0] = f1(uhL[0], uhL[1], uhL[2], gamma1);
    fL[1] = f2(uhL[0], uhL[1], uhL[2], gamma1);
    fL[2] = f3(uhL[0], uhL[1], uhL[2], gamma1);

    fRn[0] = f1(uhRn[0], uhRn[1], uhRn[2], gamma1);
    fRn[1] = f2(uhRn[0], uhRn[1], uhRn[2], gamma1);
    fRn[2] = f3(uhRn[0], uhRn[1], uhRn[2], gamma1);

    SR = fmax(abs(maxabsspeed(uhRn[0], uhRn[1], uhRn[2], gamma, gamma1)), abs(maxabsspeed(uhL[0], uhL[1], uhL[2], gamma, gamma1)));

    // The Lax-Friedrichs flux !
    for (n = 0; n < NumEq; n++)
    {
        fhatL[n] = 0.5 * (fL[n] + fRn[n]) - 0.5 * SR * (uhL[n] - uhRn[n]);
    }


    // Calculate the surface integral !
    for (d = 0; d < NumGLP; d++)
    {
        for (n = 0; n < NumEq; n++)
        {
            if (d == k)
            {
                du[i][d][n] = du[i][d][n] - (2.0 / hx) * (fhatR[n] - fint[k][n]) / weight[k];
            }
            if (d == 0)
            {
                du[i][d][n] = du[i][d][n] + (2.0 / hx) * (fhatL[n] - fint[0][n]) / weight[0];
            }
        }
    }

}


double Lx(double x, int j, double* xi, int n) {

    // calculate the derivative of j-th Lagrange function at point x.
    // n: the total point. (= NumGLP.)
    // xi(n): the nodal point vector. (= lambda.)
    // D(j,L) = L_L(x_j) = fun( lambda(j), L, lambda_ptr, NumGLP ).

    double sum = 0.0;

    for (int d = 0; d < n; d++) {
        if (d != j) {
            double product = 1.0;
            for (int i = 0; i < n; i++) {
                if (i != j && i != d) {
                    product *= (x - xi[i]) / (xi[j] - xi[i]);
                }
            }
            sum += product / (xi[j] - xi[d]);
            //printf("%d  %.16f  %d  %.16f\n", d, x, j, sum);
        }
    }
    //printf("\n");

    return sum;

}

//***************************************************************************************
/* The main function */
//***************************************************************************************

int main()
{
    double uh[Nx1 + 1][NumGLP][NumEq], ureal[Nx1 + 1][NumGLP][NumEq];
    double uh1[Nx1 + 1][NumGLP][NumEq], uh2[Nx1 + 1][NumGLP][NumEq];
    double du[Nx1 + 1][NumGLP][NumEq]; // , dutest[Nx1 + 1][NumGLP][NumEq];

    printf("Hello DG %d !\n", 2024);

    //***************************************************************************************
    /* get GLP */
    //***************************************************************************************

    if (NumGLP == 2)
    {
        lambda[0] = -1;
        lambda[1] = 1;

        weight[0] = 1;
        weight[1] = 1;
    }
    else if (NumGLP == 3)
    {
        lambda[0] = -1;
        lambda[1] = 0;
        lambda[2] = 1;

        weight[0] = 1.0 / 3;
        weight[1] = 4.0 / 3;
        weight[2] = 1.0 / 3;
    }
    else if (NumGLP == 4)
    {
        lambda[0] = -1;
        lambda[1] = -0.447213595499958;
        lambda[2] = 0.447213595499958;
        lambda[3] = 1;

        weight[0] = 1.0 / 6;
        weight[1] = 5.0 / 6;
        weight[2] = 5.0 / 6;
        weight[3] = 1.0 / 6;
    }
    else if (NumGLP == 5)
    {
        lambda[0] = -1;
        lambda[1] = -0.654653670707977;
        lambda[2] = 0;
        lambda[3] = 0.654653670707977;
        lambda[4] = 1;

        weight[0] = 0.1;
        weight[1] = 0.544444444444444;
        weight[2] = 0.711111111111111;
        weight[3] = 0.544444444444444;
        weight[4] = 0.1;
    }


    //***************************************************************************************
    /* setup initial data */
    //***************************************************************************************

    data.set_init();

    hx = (data.xb - data.xa) / Nx;
    hx1 = 0.5 * hx;

    for (i = 0; i <= Nx1; i++)
    {
        Xc[i] = data.xa + i * hx - hx1;
    }

    for (i = 0; i <= Nx1; i++)
    {
        for (i1 = 0; i1 < NumGLP; i1++)
        {
            uh[i][i1][0] = data.U1(Xc[i] + hx1 * lambda[i1]);
            ureal[i][i1][0] = data.U1(Xc[i] + hx1 * lambda[i1]);

            uh[i][i1][1] = data.U2(Xc[i] + hx1 * lambda[i1]);
            ureal[i][i1][1] = data.U2(Xc[i] + hx1 * lambda[i1]);

            uh[i][i1][2] = data.U3(Xc[i] + hx1 * lambda[i1]);
            ureal[i][i1][2] = data.U3(Xc[i] + hx1 * lambda[i1]);
        }
    }


    //***************************************************************************************
    /* get basis function */
    //***************************************************************************************

    double* lambda_ptr = (double*)malloc(NumGLP * sizeof(double));
    for (int i = 0; i < NumGLP; i++) {
        lambda_ptr[i] = lambda[i];
    }

    // calculate Dmat.
    // D(j,L) = L_L(x_j).
    for (i = 0; i < NumGLP; i++)
    {
        for (j = 0; j < NumGLP; j++)
        {
            Dmat[i][j] = Lx(lambda[i], j, lambda_ptr, NumGLP);
        }
    }

    //clock_t start_time = clock();
    auto start_time = std::chrono::high_resolution_clock::now();


    //***************************************************************************************
    /* RK3 */
    //***************************************************************************************

    double(*dev_uh)[NumGLP][NumEq], (*dev_du)[NumGLP][NumEq], (*dev_Dmat)[NumGLP], (*dev_weight);

    cudaMalloc(&dev_uh, (Nx1 + 1) * NumGLP * NumEq * sizeof(double));
    cudaMalloc(&dev_du, (Nx1 + 1) * NumGLP * NumEq * sizeof(double));

    cudaMalloc(&dev_Dmat, NumGLP * NumGLP * sizeof(double));
    cudaMalloc(&dev_weight, NumGLP * sizeof(double));

    cudaMemcpy(dev_Dmat, Dmat, NumGLP * NumGLP * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weight, weight, NumGLP * sizeof(double), cudaMemcpyHostToDevice);

    //dim3 threadsPerBlock(NT, 1, 1);
    //dim3 numBlocks(NB, 1, 1);

    t = 0;
    t1 = data.tend / frame;
    i1 = 1;

    if (plot == 1)
    {
        initgraph(1500, 700);  //initialize the screen
        setbkcolor(WHITE);      // set background color
    }

    // RK4
    while (t < data.tend)
    {
        // calculate dt
        alpha = 0;
        for (i = 0; i <= Nx1; i++)
        {
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                alpha = fmax(fmax(alpha, abs(minspeed(uh[i][i1][0], uh[i][i1][1], uh[i][i1][2], gamma, gamma1))), abs(maxspeed(uh[i][i1][0], uh[i][i1][1], uh[i][i1][2], gamma, gamma1)));
            }
        }

        dt = CFL * hx / alpha;

        if (t + dt >= data.tend)
        {
            dt = data.tend - t;
            t = data.tend;
        }
        else
        {
            t = t + dt;
        }

        for (i = 0; i <= Nx1; i++)
        {
            for (d = 0; d < NumGLP; d++)
            {
                for (n = 0; n < NumEq; n++)
                {
                    uh1[i][d][n] = uh[i][d][n];
                    uh2[i][d][n] = uh[i][d][n];
                }
            }
        }

        // Stage I
        for (int I = 1; I <= 5; I++)
        {
            set_bc(uh1);
            cudaMemcpy(dev_uh, uh1, (Nx1 + 1) * NumGLP * NumEq * sizeof(double), cudaMemcpyHostToDevice);
            Lh << <NB, NT >> > (dev_uh, dev_du, hx, dev_weight, dev_Dmat, gamma, gamma1);
            cudaMemcpy(du, dev_du, (Nx1 + 1) * NumGLP * NumEq * sizeof(double), cudaMemcpyDeviceToHost);

            for (i = 0; i <= Nx1; i++)
            {
                for (d = 0; d < NumGLP; d++)
                {
                    for (n = 0; n < NumEq; n++)
                    {
                        uh1[i][d][n] = uh1[i][d][n] + (dt / 6) * du[i][d][n];
                    }
                }
            }
            Limiter(uh1, dt);
        }

        for (i = 0; i <= Nx1; i++)
        {
            for (d = 0; d < NumGLP; d++)
            {
                for (n = 0; n < NumEq; n++)
                {
                    uh2[i][d][n] = 0.04 * uh2[i][d][n] + 0.36 * uh1[i][d][n];
                    uh1[i][d][n] = 15 * uh2[i][d][n] - 5 * uh1[i][d][n];
                }
            }
        }

        // Stage II
        for (int I = 6; I <= 9; I++)
        {
            set_bc(uh1);
            cudaMemcpy(dev_uh, uh1, (Nx1 + 1) * NumGLP * NumEq * sizeof(double), cudaMemcpyHostToDevice);
            Lh << <NB, NT >> > (dev_uh, dev_du, hx, dev_weight, dev_Dmat, gamma, gamma1);
            cudaMemcpy(du, dev_du, (Nx1 + 1) * NumGLP * NumEq * sizeof(double), cudaMemcpyDeviceToHost);

            for (i = 0; i <= Nx1; i++)
            {
                for (d = 0; d < NumGLP; d++)
                {
                    for (n = 0; n < NumEq; n++)
                    {
                        uh1[i][d][n] = uh1[i][d][n] + (dt / 6) * du[i][d][n];
                    }
                }
            }
            Limiter(uh1, dt);
        }

        // Stage III
        set_bc(uh1);
        cudaMemcpy(dev_uh, uh1, (Nx1 + 1) * NumGLP * NumEq * sizeof(double), cudaMemcpyHostToDevice);
        Lh << <NB, NT >> > (dev_uh, dev_du, hx, dev_weight, dev_Dmat, gamma, gamma1);
        cudaMemcpy(du, dev_du, (Nx1 + 1) * NumGLP * NumEq * sizeof(double), cudaMemcpyDeviceToHost);

        for (i = 0; i <= Nx1; i++)
        {
            for (d = 0; d < NumGLP; d++)
            {
                for (n = 0; n < NumEq; n++)
                {
                    uh[i][d][n] = uh2[i][d][n] + 0.6 * uh1[i][d][n] + 0.1 * dt * du[i][d][n];
                }
            }
        }
        Limiter(uh, dt);

        // calculate umax and umin
        uumax = uh[1][0][0];
        uumin = uh[1][0][0];
        for (i = 0; i <= Nx1; i++)
        {
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                uumax = fmax(uumax, uh[i][i1][0]);
                uumin = fmin(uumin, uh[i][i1][0]);
            }
        }

        printf("% .16E % .16E % .16E\n", t, uumax, uumin);

        if (t >= ii1 * t1)
        {
            if (plot == 1)
            {
                cleardevice();       // clear
                setlinecolor(RED);    // set axis color
                setorigin(500, 350);    // set (0,0)
                line(-500, 00, 1000, 00);    // plot x-axis
                line(0, 350, 0, -350);  // plot y-axis
                setlinecolor(BLACK);
                // draw the polynomial
                for (i = 1; i <= Nx; i++)
                {
                    for (i1 = 1; i1 < NumGLP; i1++)
                    {
                        line(scalex * (Xc[i] + hx1 * lambda[i1 - 1]), -scaley * uh[i][i1 - 1][0], scalex * (Xc[i] + hx1 * lambda[i1]), -scaley * uh[i][i1][0]);
                    }
                }
                // draw the cell-average
                for (i = 2; i <= Nx; i++)
                {
                    //    line(scalex * Xc[i - 1], -scaley * uh[i - 1][0], scalex * Xc[i], -scaley * uh[i][0]);
                }
                TCHAR str_time[50];
                TCHAR str_umax[50];
                TCHAR str_umin[50];
                settextcolor(BLACK);
                _stprintf_s(str_time, 50, _T("t = %.16f"), t);
                _stprintf_s(str_umax, 50, _T("The maximal value of rho = %.16f"), uumax);
                _stprintf_s(str_umin, 50, _T("The minimal value of rho = %.16f"), uumin);
                outtextxy(scalex * 5, -scaley * (-1), str_time);
                outtextxy(scalex * 5, -scaley * (-2), str_umax);
                outtextxy(scalex * 5, -scaley * (-3), str_umin);
                ii1 = ii1 + 1;
                Sleep(1);
            }
        }

    }







    //***************************************************************************************
    /* calculate_L2_error */
    //***************************************************************************************

    for (i = 1; i <= Nx; i++)
    {
        for (i1 = 0; i1 < NumGLP; i1++)
        {
            uE = abs(uh[i][i1][0] - ureal[i][i1][0]);
            L1_Error = L1_Error + 0.5 * weight[i1] * uE;
            L2_Error = L2_Error + 0.5 * weight[i1] * pow(uE, 2);
            L8_Error = fmax(L8_Error, uE);
        }
    }

    L1_Error = L1_Error / Nx;
    L2_Error = pow(L2_Error / Nx, 0.5);

    printf("\n");
    printf("The Error:\n");
    printf("L1 Error: %.16E\n", L1_Error);
    printf("L2 Error: %.16E\n", L2_Error);
    printf("L8 Error: %.16E\n", L8_Error);
    printf("\n");
    printf("The computation is over !\n");

    //clock_t end_time = clock();
    auto end_time = std::chrono::high_resolution_clock::now();

    //double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    std::chrono::duration<double> time_taken = end_time - start_time;

    //printf("Time use is %.6f seconds ! \n ", time_taken);
    printf("Time use is %.6f seconds ! \n ", time_taken.count());

    printf("Save the solution ...\n");

    FILE* Q = fopen("Q.txt", "w");
    FILE* Xcfile = fopen("Xc.txt", "w");

    double uhbar[Nx1 + 1][NumEq];

    // calculate uhbar
    for (i = 1; i <= Nx; i++)
    {
        for (n = 0; n < NumEq; n++)
        {
            uhbar[i][n] = 0;
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                uhbar[i][n] = uhbar[i][n] + 0.5 * weight[i1] * uh[i][i1][n];
            }
        }
    }

    // 写入矩阵数据
    for (i = 1; i <= Nx; i++)
    {
        for (n = 0; n < NumEq; n++)
        {
            fprintf(Q, "%.16E\n", uhbar[i][n]);
        }
        fprintf(Xcfile, "%.16E\n", Xc[i]);
    }

    // 关闭文件
    fclose(Q);
    fclose(Xcfile);

    printf("Complete !\n");

    if (plot == 1)
    {
    TCHAR str[50];
    settextcolor(RED);
    _stprintf_s(str, 50, _T("Complete !"));
    outtextxy(scalex*5, -scaley*(-5), str);
    }

    system("pause");

    closegraph();

    return 0;
}

