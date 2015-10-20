/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% PETSc behind the scenes maintenance functions
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#include "mycalls.hpp"
#include <stdlib.h>
#include <iostream>

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
    
//    ierr = VecDuplicate(obj->rhs, &obj->current_temperature_field_local);
//    ierr = VecCreate(PETSC_COMM_WORLD, &obj->current_temperature_field_local);
//    ierr = VecSetSizes(obj->current_temperature_field_local, PETSC_DECIDE, m);
//    ierr = VecSetFromOptions(obj->current_temperature_field_local);

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
    /*
    ierr = MatDuplicate(obj->Amat, MAT_COPY_VALUES, &obj->mass_matrix);
    ierr = MatDuplicate(obj->Amat, MAT_COPY_VALUES, &obj->stiffness_matrix);
    */
    ierr = MatCreate(PETSC_COMM_WORLD, &obj->stiffness_matrix);
    ierr = MatSetSizes(obj->stiffness_matrix,PETSC_DECIDE,PETSC_DECIDE,m,n);
    ierr = MatSetFromOptions(obj->stiffness_matrix);
    
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
 %% Function to do final assembly of matrices
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_Assem_Matrices(PETSC_STRUCT *obj)
{
    PetscErrorCode ierr;
    ierr = MatAssemblyBegin(obj->Amat, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(obj->Amat, MAT_FINAL_ASSEMBLY);
    
    ierr = MatAssemblyBegin(obj->stiffness_matrix, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(obj->stiffness_matrix, MAT_FINAL_ASSEMBLY);
    
    //Indicate same nonzero structure of successive linear system matrices
    MatSetOption(obj->Amat, MAT_NO_NEW_NONZERO_LOCATIONS);
    MatSetOption(obj->stiffness_matrix, MAT_NO_NEW_NONZERO_LOCATIONS);
    
    MatSetOption(obj->Amat, MAT_SYMMETRIC);
    MatSetOption(obj->stiffness_matrix, MAT_SYMMETRIC);

    return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to do final assembly of vectors
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_Assem_Vectors(PETSC_STRUCT *obj)
{
    PetscErrorCode ierr;
    ierr = VecAssemblyBegin(obj->rhs);
    ierr = VecAssemblyEnd(obj->rhs);
    
    ierr = VecAssemblyBegin(obj->current_temperature_field_local);
    ierr = VecAssemblyEnd(obj->current_temperature_field_local);
//    ierr = VecAssemblyBegin(obj->initial_temperature_field);
//    ierr = VecAssemblyEnd(obj->initial_temperature_field);

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
    
//    s
//    ierr = VecDestroy(obj->initial_temperature_field);
//    ierr = MatDestroy(obj->mass_matrix);
    ierr = MatDestroy(obj->stiffness_matrix);

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
