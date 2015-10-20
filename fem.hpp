/*** Header file ***/
/*** Include standard libraries ***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mycalls.hpp"
/*** Global variables ***
 * b = left side of the tri-diagonal system Ax=b *
 * x = xpoints or nodal values *
 * dlen = length of the intervals between adjacent nodes *
 * qdwt = quadrature weights *
 * xquadpt = points used in Simpson’s rule for numerical *
 * integration of each interval *
 * atri = array to hold the tri-diagonal A matrix in the *
 * system Ax = b *
 * ni_indx = array to hold the location of the endpoints of *
 * each interval *
 * indx = array to hold the offset between interval number *
 * and node number *
 ********************************************************************/
/*** Function Prototypes ***/
/*
void bookkeep(int n);
void assem(PETSC_STRUCT *obj, int n);
void errors(double *el2,double *emax,int n);
double exact(double x);
double f(double z);
double p(double z);
double q(double z);
double bspl(double xx,int intrvl,int ij,int id);

#ifndef _FIX
extern double b[100],x[100],dlen[100],qdwt[3],xquadpt[100][5],atri[100][3],
    xpts[100],xleft,xright,dintrvl,add_term,
    basisi,drvi,basisj,drvj,aij,wght,test,testprime,xx;
extern int ni_indx[100][2],indx[100],i,j,ileft,iright,intrvl,jj,
    jbasis,jendpt,iquad,ibasis;
#endif
*/