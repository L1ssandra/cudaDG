#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include<graphics.h>
#include<conio.h>
#include<stdlib.h>

//***************************************************************************************
/* Parameters */
//***************************************************************************************

int const Nx = 200;
int const k = 2;
double const CFL = 0.2;
int const RKorder = 3;
int const NumGLP = 5;
int const plot = 1;

// CUDA data !
int const NB = 25; // Number of block

// draw the picture
int const frame = 400;
double const scalex = 150;
double const scaley = 150;


int const dimPk = k + 1;
int const Nx1 = Nx + 1;
double const pi = 4 * atan(1.0);
int const NT = Nx / NB; // Number of threads in each block


/*
Examples:
1: sin(x) in [0,2*pi]
2: step function in [0,2*pi]
3: 0.5 + sin(x) in [0,2*pi]
*/
int const example = 3;


/*
Equations:
1: linear
2: burgers
*/
int const equ = 2;


// The numerical solution
double hx, hx1;
double Xc[Nx1 + 1];
double t, t1, dt, alpha;
double L1_Error = 0, L2_Error = 0, L8_Error = 0, uE;
double uhG[Nx1 + 1][NumGLP], uhR[Nx1 + 1], uhL[Nx1 + 1];
double uumax, uumin;


// The basis
double phiG[NumGLP][dimPk], phixG[NumGLP][dimPk];
double phiGR[dimPk], phiGL[dimPk];
double lambda[NumGLP], weight[NumGLP];
double mm[dimPk];


// The loop variables
int i, i1, d, ii1;

//***************************************************************************************
/* The flux function */
//***************************************************************************************
class equation // f(u) = u
{
public:
    __host__ __device__ double f(double u)
    {
        if (equ == 1)
        {
            return u;
        }
        else if (equ == 2)
        {
            return pow(u, 2) / 2;
        }
    }
    __host__ __device__ double df(double u)
    {
        if (equ == 1)
        {
            return 1;
        }
        else if (equ == 2)
        {
            return u;
        }
    }
};
equation HCL;

//***************************************************************************************
/* The initial data */
//***************************************************************************************
class init
{
public:

    double const pi = 4 * atan(1.0);
    double xa, xb, tend;
    int bcL, bcR;

    __host__ __device__ void set_init()
    {

        if (example == 1 || example == 2)
        {
            xa = 0;
            xb = 2 * pi;
            bcL = 1;
            bcR = 1;
            tend = 2 * pi;
        }
        else if (example == 3)
        {
            xa = 0;
            xb = 2 * pi;
            bcL = 1;
            bcR = 1;
            tend = 2.2;
        }
    }

    double u0(double x)
    {
        if (example == 1)
        {
            return sin(x);
        }
        else if (example == 2)
        {
            if (x < pi / 2 || x > 3 * pi / 2)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
        else if (example == 3)
        {
            return 0.5 + sin(x);
        }
    }

};
init data;



void set_bc(double uh[Nx1 + 1][dimPk])
{

    // Set boundary condition !
    if (data.bcL == 1)
    {
        for (d = 0; d < dimPk; d++)
        {
            uh[0][d] = uh[Nx][d];
        }
    }

    if (data.bcR == 1)
    {
        for (d = 0; d < dimPk; d++)
        {
            uh[Nx1][d] = uh[1][d];
        }
    }

}

//***************************************************************************************
/* Discontinuous Galerkin space discretization */
//***************************************************************************************

void LhCPU(double uh[Nx1 + 1][dimPk], double du[Nx1 + 1][dimPk])
{
    double fL, fR, uL, uR, SR, SL;
    double fhat[Nx1];
    double uhG[Nx1 + 1][NumGLP], uhR[Nx1 + 1], uhL[Nx1 + 1];

    // Compute the value of uh(x) at Gauss points: uhG[0:Nx1,0:NumGLP - 1] !
    for (i = 1; i <= Nx; i++)
    {
        for (i1 = 0; i1 < NumGLP; i1++)
        {
            uhG[i][i1] = 0;
            for (d = 0; d < dimPk; d++)
            {
                uhG[i][i1] = uhG[i][i1] + uh[i][d] * phiG[i1][d];
            }
        }
    }

    // Calculate the volume integral: \int_Ii f(uh)*phi' dx !
    for (i = 1; i <= Nx; i++)
    {
        for (d = 0; d < dimPk; d++)
        {
            du[i][d] = 0;
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                du[i][d] = du[i][d] + 0.5 * weight[i1] * HCL.f(uhG[i][i1]) * phixG[i1][d];
            }
        }
    }


    // Calculate the surface values: uhR[1:Nx], uhL[1:Nx] !
    for (i = 1; i <= Nx; i++)
    {
        uhR[i] = 0;
        uhL[i] = 0;
        for (d = 0; d < dimPk; d++)
        {
            uhR[i] = uhR[i] + uh[i][d] * phiGR[d];
            uhL[i] = uhL[i] + uh[i][d] * phiGL[d];
        }
    }

    // Set boundary condition !
    if (data.bcL == 1)
    {
        uhR[0] = uhR[Nx];
    }

    if (data.bcR == 1)
    {
        uhL[Nx1] = uhL[1];
    }

    // Calculate numerical flux: fhat[0:Nx] !
    for (i = 0; i <= Nx; i++)
    {
        uR = uhL[i + 1];
        uL = uhR[i];
        fR = HCL.f(uR);
        fL = HCL.f(uL);
        SR = fmax(abs(HCL.df(uR)), abs(HCL.df(uL)));

        // The Lax-Friedrichs flux !
        fhat[i] = 0.5 * (fR + fL) - 0.5 * SR * (uR - uL);

        //printf("%.16f %.16f %.16f\n", fR, fL, fhat[i]);
    }

    // Calculate the surface integral !
    for (i = 1; i <= Nx; i++)
    {
        if (i == 25)
        {
            printf("CPU: %.16f  %.16f\n", fhat[i], fhat[i - 1]);
        }
        for (d = 0; d < dimPk; d++)
        {
            //du[i][d] = du[i][d] - (1 * phiGR[d] - 1 * phiGL[d]) / hx;
            du[i][d] = du[i][d] - (fhat[i] * phiGR[d] - fhat[i - 1] * phiGL[d]) / hx;
        }
    }

    // divide the mass
    for (i = 1; i <= Nx; i++)
    {
        for (d = 0; d < dimPk; d++)
        {
            du[i][d] = du[i][d] / mm[d];
        }
    }
}

__global__ void Lh(double uh[Nx1 + 1][dimPk], double du[Nx1 + 1][dimPk], double phiG[NumGLP][dimPk], double phixG[NumGLP][dimPk], double hx, double mm[dimPk], double weight[NumGLP], double phiGR[dimPk], double phiGL[dimPk])
{
    double fL, fR, uL, uR, SR, SL;
    double fhatR, fhatL;
    double uhG[Nx1 + 1][NumGLP]; // , uhR[Nx1 + 1], uhL[Nx1 + 1];
    double uhR, uhRn, uhLp, uhL;
    int i, i1, d;

    equation HCL;
    init data;
    data.set_init();

    i = blockIdx.x * blockDim.x + threadIdx.x + 1; // i from 1 to Nx

    // Compute the value of uh(x) at Gauss points: uhG[0:Nx1,0:NumGLP - 1] !
    for (i1 = 0; i1 < NumGLP; i1++)
    {
        uhG[i][i1] = 0;
        for (d = 0; d < dimPk; d++)
        {
            uhG[i][i1] = uhG[i][i1] + uh[i][d] * phiG[i1][d];
        }
    }


    // Calculate the volume integral: \int_Ii f(uh)*phi' dx !
    for (d = 0; d < dimPk; d++)
    {
        du[i][d] = 0;
        for (i1 = 0; i1 < NumGLP; i1++)
        {
            du[i][d] = du[i][d] + 0.5 * weight[i1] * HCL.f(uhG[i][i1]) * phixG[i1][d];
        }
    }


    // Calculate the surface values: uhR[1:Nx], uhL[1:Nx] !
    uhRn = 0;
    uhR = 0;
    uhL = 0;
    uhLp = 0;
    for (d = 0; d < dimPk; d++)
    {
        uhR = uhR + uh[i][d] * phiGR[d];
        uhL = uhL + uh[i][d] * phiGL[d];
        uhLp = uhLp + uh[i + 1][d] * phiGL[d];
        uhRn = uhRn + uh[i - 1][d] * phiGR[d];
    }

    // Calculate numerical flux: fhat[0:Nx] !
    uR = uhLp;
    uL = uhR;
    fR = HCL.f(uR);
    fL = HCL.f(uL);
    SR = fmax(abs(HCL.df(uR)), abs(HCL.df(uL)));

    // The Lax-Friedrichs flux !
    fhatR = 0.5 * (fR + fL) - 0.5 * SR * (uR - uL);


    uR = uhL;
    uL = uhRn;
    fR = HCL.f(uR);
    fL = HCL.f(uL);
    SR = fmax(abs(HCL.df(uR)), abs(HCL.df(uL)));

    // The Lax-Friedrichs flux !
    fhatL = 0.5 * (fR + fL) - 0.5 * SR * (uR - uL);


    // Calculate the surface integral !
    for (d = 0; d < dimPk; d++)
    {
        //du[i][d] = du[i][d] - (1 * phiGR[d] - 1 * phiGL[d]) / hx;
        du[i][d] = du[i][d] - (fhatR * phiGR[d] - fhatL * phiGL[d]) / hx;
        //du[i][d] = du[i][d] - (uhR * phiGR[d] - uhL * phiGL[d]) / hx;

        //printf("%.16f  %.16f\n", phiGR[d], phiGL[d]);
    }

    // divide the mass
    for (d = 0; d < dimPk; d++)
    {
        du[i][d] = du[i][d] / mm[d];
    }
    
}


//***************************************************************************************
/* The main function */
//***************************************************************************************

int main()
{
    double uh[Nx1 + 1][dimPk], ureal[Nx1 + 1][NumGLP];
    double uh1[Nx1 + 1][dimPk], uh2[Nx1 + 1][dimPk];
    double du[Nx1 + 1][dimPk], dutest[Nx1 + 1][dimPk];

    printf("Hello DG %d !\n", 2024);


    //***************************************************************************************
    /* get GLP */
    //***************************************************************************************

    if (NumGLP == 5)
    {
        lambda[0] = -0.9061798459386639927976269;
        lambda[1] = -0.5384693101056830910363144;
        lambda[2] = 0;
        lambda[3] = 0.5384693101056830910363144;
        lambda[4] = 0.9061798459386639927976269;

        weight[0] = 0.2369268850561890875142640;
        weight[1] = 0.4786286704993664680412915;
        weight[2] = 0.5688888888888888888888889;
        weight[3] = 0.4786286704993664680412915;
        weight[4] = 0.2369268850561890875142640;
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
            ureal[i][i1] = data.u0(Xc[i] + hx1 * lambda[i1]);
        }
    }


    //***************************************************************************************
    /* get basis function */
    //***************************************************************************************

    for (i = 0; i < NumGLP; i++)
    {
        phiG[i][0] = 1;
        phiG[i][1] = lambda[i];
        phiG[i][2] = pow(lambda[i], 2) - 1.0 / 3;

        phixG[i][0] = 0;
        phixG[i][1] = 1.0 / hx1;
        phixG[i][2] = 2.0 * lambda[i] / hx1;
    }

    phiGR[0] = 1;
    phiGR[1] = 1;
    phiGR[2] = 2.0 / 3;

    phiGL[0] = 1;
    phiGL[1] = -1;
    phiGL[2] = 2.0 / 3;

    mm[0] = 1;
    mm[1] = 1.0 / 3;
    mm[2] = 4.0 / 45;


    //***************************************************************************************
    /* The L2 projection */
    //***************************************************************************************

    for (i = 0; i <= Nx1; i++)
    {
        for (d = 0; d < dimPk; d++)
        {
            uh[i][d] = 0;
            for (i1 = 0; i1 < NumGLP; i1++)
            {
                uh[i][d] = uh[i][d] + 0.5 * weight[i1] * ureal[i][i1] * phiG[i1][d] / mm[d];
            }
        }
    }


    //***************************************************************************************
    /* RK3 */
    //***************************************************************************************

    double(*dev_uh)[dimPk], (*dev_du)[dimPk], (*dev_phiG)[dimPk], (*dev_phixG)[dimPk], (*dev_mm), (*dev_weight);
    double(*dev_phiGR), (*dev_phiGL);

    cudaMalloc(&dev_uh, (Nx1 + 1) * dimPk * sizeof(double));
    cudaMalloc(&dev_du, (Nx1 + 1) * dimPk * sizeof(double));

    cudaMalloc(&dev_phiG, NumGLP * dimPk * sizeof(double));
    cudaMalloc(&dev_phixG, NumGLP * dimPk * sizeof(double));
    cudaMalloc(&dev_mm, dimPk * sizeof(double));
    cudaMalloc(&dev_weight, NumGLP * sizeof(double));
    cudaMalloc(&dev_phiGR, dimPk * sizeof(double));
    cudaMalloc(&dev_phiGL, dimPk * sizeof(double));

    cudaMemcpy(dev_phiG, phiG, NumGLP * dimPk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_phixG, phixG, NumGLP * dimPk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mm, mm, dimPk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weight, weight, NumGLP * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_phiGR, phiGR, dimPk * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_phiGL, phiGL, dimPk * sizeof(double), cudaMemcpyHostToDevice);

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

    while (t < data.tend)
    {
        // calculate dt
        alpha = 0;
        for (i = 0; i <= Nx1; i++)
        {
            alpha = fmax(alpha, HCL.df(uh[i][0]));
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

        // Stage I
        set_bc(uh);
        cudaMemcpy(dev_uh, uh, (Nx1 + 1) * dimPk * sizeof(double), cudaMemcpyHostToDevice);
        Lh << <NB, NT >> > (dev_uh, dev_du, dev_phiG, dev_phixG, hx, dev_mm, dev_weight, dev_phiGR, dev_phiGL);
        cudaMemcpy(du, dev_du, (Nx1 + 1) * dimPk * sizeof(double), cudaMemcpyDeviceToHost);
        //LhCPU(uh, dutest);

        for (i = 0; i <= Nx1; i++)
        {
            for (d = 0; d < dimPk; d++)

            {
                uh1[i][d] = uh[i][d] + dt * du[i][d];
            }
        }
        
        // Stage II
        set_bc(uh1);
        cudaMemcpy(dev_uh, uh1, (Nx1 + 1)* dimPk * sizeof(double), cudaMemcpyHostToDevice);
        Lh << <NB, NT >> > (dev_uh, dev_du, dev_phiG, dev_phixG, hx, dev_mm, dev_weight, dev_phiGR, dev_phiGL);
        cudaMemcpy(du, dev_du, (Nx1 + 1)* dimPk * sizeof(double), cudaMemcpyDeviceToHost);
        //LhCPU(uh1, du);

        for (i = 0; i <= Nx1; i++)
        {
            for (d = 0; d < dimPk; d++)
            {
                uh2[i][d] = (3.0 / 4) * uh[i][d] + (1.0 / 4) * uh1[i][d] + (1.0 / 4) * dt * du[i][d];
            }
        }

        // Stage III
        set_bc(uh2);
        cudaMemcpy(dev_uh, uh2, (Nx1 + 1) * dimPk * sizeof(double), cudaMemcpyHostToDevice);
        Lh << <NB, NT >> > (dev_uh, dev_du, dev_phiG, dev_phixG, hx, dev_mm, dev_weight, dev_phiGR, dev_phiGL);
        cudaMemcpy(du, dev_du, (Nx1 + 1) * dimPk * sizeof(double), cudaMemcpyDeviceToHost);
        //LhCPU(uh2, du);

        for (i = 0; i <= Nx1; i++)
        {
            for (d = 0; d < dimPk; d++)
            {
                uh[i][d] = (1.0 / 3) * uh[i][d] + (2.0 / 3) * uh2[i][d] + (2.0 / 3) * dt * du[i][d];
            }
        }
        
        // calculate umax and umin
        uumax = uh[1][0];
        uumin = uh[1][0];
        for (i = 0; i <= Nx1; i++)
        {
            uumax = fmax(uumax, uh[i][0]);
            uumin = fmin(uumin, uh[i][0]);
        }

        printf("% .16E % .16E % .16E\n", t, uumax, uumin);

        if (t >= ii1 * t1)
        {
            if (plot == 1)
            {
                cleardevice();       // clear
                setlinecolor(RED);    // set axis color
                setorigin(100, 350);    // set (0,0)
                line(-100, 00, 1400, 00);    // plot x-axis
                line(0, 350, 0, -350);  // plot y-axis
                setlinecolor(BLACK);
                for (i = 2; i <= Nx; i++)
                {
                    line(scalex * Xc[i - 1], -scaley * uh[i - 1][0], scalex * Xc[i], -scaley * uh[i][0]);
                }
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
            uhG[i][i1] = 0;
            for (d = 0; d < dimPk; d++)
            {
                uhG[i][i1] = uhG[i][i1] + uh[i][d] * phiG[i1][d];
            }
        }
    }

    for (i = 1; i <= Nx; i++)
    {
        for (i1 = 0; i1 < NumGLP; i1++)
        {
            uE = abs(uhG[i][i1] - ureal[i][i1]);
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

    system("pause");

    closegraph();

    return 0;
}
