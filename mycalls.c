/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% PETSc behind the scenes maintenance functions
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#include "mycalls.h"
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to initialize Petsc
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
void Petsc_Init(int argc, char **args,char *help)
{
    PetscErrorCode ierr;
    PetscInt n;
    PetscMPIInt size;
    PetscInitialize(&argc,&args,(char *)0,help);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
    ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to finalize Petsc
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_End()
{
    PetscErrorCode ierr;
    ierr = PetscFinalize();
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to create the rhs and soln vectors
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Vec_Create(PETSC_STRUCT *obj, PetscInt m)
{
    PetscErrorCode ierr;
    ierr = VecCreate(PETSC_COMM_WORLD, &obj->rhs);
    ierr = VecSetSizes(obj->rhs, PETSC_DECIDE, m);
    ierr = VecSetFromOptions(obj->rhs);
    ierr = VecDuplicate(obj->rhs, &obj->sol);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to create the system matrix
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Mat_Create(PETSC_STRUCT *obj, PetscInt m, PetscInt n)
{
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD, &obj->Amat);
    ierr = MatSetSizes(obj->Amat,PETSC_DECIDE,PETSC_DECIDE,m,n);
    ierr = MatSetFromOptions(obj->Amat);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to solve a linear system using KSP
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_Solve(PETSC_STRUCT *obj)
{
    PetscErrorCode ierr;
    ierr = KSPCreate(PETSC_COMM_WORLD,&obj->ksp);
    ierr = KSPSetOperators(obj->ksp,obj->Amat,obj->Amat, DIFFERENT_NONZERO_PATTERN);
    ierr = KSPGetPC(obj->ksp,&obj->pc);
    ierr = PCSetType(obj->pc,PCNONE);
    ierr = KSPSetTolerances(obj->ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    ierr = KSPSetFromOptions(obj->ksp);
    ierr = KSPSolve(obj->ksp,obj->rhs,obj->sol);
    ierr = VecAssemblyBegin(obj->sol);
    ierr = VecAssemblyEnd(obj->sol);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to do final assembly of matrix and right hand side vector
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_Assem(PETSC_STRUCT *obj)
{
    PetscErrorCode ierr;
    ierr = MatAssemblyBegin(obj->Amat, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(obj->Amat, MAT_FINAL_ASSEMBLY);
    ierr = VecAssemblyBegin(obj->rhs);
    ierr = VecAssemblyEnd(obj->rhs);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to Destroy the matrix and vectors that have been created
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_Destroy(PETSC_STRUCT *obj)
{
    PetscErrorCode ierr;
    ierr = VecDestroy(obj->rhs);
    ierr = VecDestroy(obj->sol);
    ierr = MatDestroy(obj->Amat);
    ierr = KSPDestroy(obj->ksp);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to View the matrix and vectors that have been created in an m-file
 %% Note: Assumes all final assemblies of matrices and vectors have been performed
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_View(PETSC_STRUCT obj, PetscViewer viewer)
{
    PetscErrorCode ierr;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "results.m", &viewer);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
    ierr = PetscObjectSetName((PetscObject)obj.Amat,"Amat");
    ierr = PetscObjectSetName((PetscObject)obj.rhs,"rhs");
    ierr = PetscObjectSetName((PetscObject)obj.sol,"sol");
    ierr = MatView(obj.Amat,viewer);
    ierr = VecView(obj.rhs, viewer);
    ierr = VecView(obj.sol, viewer);
    return;
}
