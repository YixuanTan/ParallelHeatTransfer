#include "petscmat.h"
#include "petscksp.h"
typedef Mat PETSC_MAT;
typedef Vec PETSC_VEC;
typedef struct
{
    PETSC_MAT Amat; /*linear system matrix*/
    KSP ksp; /*linear solver context*/
    PC pc; /*preconditioner context*/
    PETSC_VEC rhs; /*petsc rhs vector*/
    PETSC_VEC sol; /*petsc solution vector*/
    
    PETSC_MAT stiffness_matrix;
    PETSC_VEC current_temperature_field_local;
}PETSC_STRUCT;

