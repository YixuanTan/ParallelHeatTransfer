/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% PETSc behind the scenes maintenance functions
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#include "mycalls.hpp"
#include <stdlib.h>
#include <iostream>
#include "petscmat.h"
#include <petscksp.h>

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to initialize Petsc
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
void Petsc_Init(int argc, char **args,char *help)
{
    PetscErrorCode ierr;
    PetscInt n;
    PetscMPIInt size;
    PetscInitialize(&argc,&args,(char *)0,help);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); //CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL); //CHKERRQ(ierr);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to finalize Petsc
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_End()
{
    PetscErrorCode ierr;
    ierr = PetscFinalize(); //CHKERRQ(ierr);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to create the rhs and soln vectors
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Vec_Create(PETSC_STRUCT *obj, PetscInt m)
{
    PetscErrorCode ierr;
    ierr = VecCreate(PETSC_COMM_WORLD, &obj->rhs); //CHKERRQ(ierr);
    ierr = VecSetSizes(obj->rhs, PETSC_DECIDE, m); //CHKERRQ(ierr);
    ierr = VecSetFromOptions(obj->rhs); //CHKERRQ(ierr);
    ierr = VecDuplicate(obj->rhs, &obj->sol); //CHKERRQ(ierr);
    
   ierr = VecDuplicate(obj->rhs, &obj->current_temperature_field_local);
//    ierr = VecCreate(PETSC_COMM_WORLD, &obj->current_temperature_field_local);
//    ierr = VecSetSizes(obj->current_temperature_field_local, PETSC_DECIDE, m);
//    ierr = VecSetFromOptions(obj->current_temperature_field_local);
    
//    ierr = VecSetOption(obj->rhs, VEC_IGNORE_OFF_PROC_ENTRIES); // assembling rhs may need info from other rank
//    ierr = VecSetOption(obj->sol, VEC_IGNORE_OFF_PROC_ENTRIES);
    ierr = VecSetOption(obj->current_temperature_field_local, VEC_IGNORE_OFF_PROC_ENTRIES);
    
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
    //ierr = MatSetFromOptions(obj->Amat);
    ierr = MatSetType(obj->Amat, MATMPIAIJ);
    // d_nz <= 9  o_nz <=8 (at least one at the diagonal)
//    ierr = MatMPIAIJSetPreallocation(obj->Amat, 9, d_nnz, 8, o_nnz);
    
    //    MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, 9, PETSC_NULL, 9, PETSC_NULL, obj->Amat);

//    ierr = MatDuplicate(obj->Amat, MAT_DO_NOT_COPY_VALUES, &obj->stiffness_matrix);

    ierr = MatCreate(PETSC_COMM_WORLD, &obj->stiffness_matrix);
    ierr = MatSetSizes(obj->stiffness_matrix,PETSC_DECIDE,PETSC_DECIDE,m,n);
//    ierr = MatSetFromOptions(obj->stiffness_matrix);
    ierr = MatSetType(obj->stiffness_matrix, MATMPIAIJ);
    // d_nz <= 9  o_nz <=8 (at least one at the diagonal)
//    ierr = MatMPIAIJSetPreallocation(obj->stiffness_matrix, 9, d_nnz, 8, o_nnz);

    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to create d_nnz and o_nnz
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Mat_Preallocation(PETSC_STRUCT *obj, PetscInt d_nnz[], PetscInt o_nnz[])
{
    PetscErrorCode ierr;
    // d_nz <= 9  o_nz <=8 (at least one at the diagonal)
    ierr = MatMPIAIJSetPreallocation(obj->Amat, 9, d_nnz, 8, o_nnz);
    
    // d_nz <= 9  o_nz <=8 (at least one at the diagonal)
    ierr = MatMPIAIJSetPreallocation(obj->stiffness_matrix, 9, d_nnz, 8, o_nnz);
    
    return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to solve a linear system using KSP
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_Solve(PETSC_STRUCT *obj)
{
    PetscErrorCode ierr;
    ierr = KSPCreate(PETSC_COMM_WORLD,&obj->ksp); //CHKERRQ(ierr);
    
    // DIFFERENT_NONZERO_PATTERN or SAME_NONZERO_PATTERN does not change the solution. TESTED.
    ierr = KSPSetOperators(obj->ksp,obj->Amat,obj->Amat, SAME_NONZERO_PATTERN); //CHKERRQ(ierr);
    ierr = KSPGetPC(obj->ksp,&obj->pc); //CHKERRQ(ierr);
//    ierr = PCSetType(obj->pc,PCNONE); //CHKERRQ(ierr);
    ierr = PCSetType(obj->pc,PCBJACOBI);
    ierr = KSPSetTolerances(obj->ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT); //CHKERRQ(ierr);
    ierr = KSPSetFromOptions(obj->ksp); //CHKERRQ(ierr);
    ierr = KSPSolve(obj->ksp,obj->rhs,obj->sol); //CHKERRQ(ierr);
    ierr = VecAssemblyBegin(obj->sol); //CHKERRQ(ierr);
    ierr = VecAssemblyEnd(obj->sol); //CHKERRQ(ierr);
    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to do final assembly of matrices
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_Assem_Matrices(PETSC_STRUCT *obj)
{
    PetscErrorCode ierr;
    ierr = MatAssemblyBegin(obj->Amat, MAT_FINAL_ASSEMBLY); //CHKERRQ(ierr);
    ierr = MatAssemblyEnd(obj->Amat, MAT_FINAL_ASSEMBLY); //CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(obj->stiffness_matrix, MAT_FINAL_ASSEMBLY); //CHKERRQ(ierr);
    ierr = MatAssemblyEnd(obj->stiffness_matrix, MAT_FINAL_ASSEMBLY); //CHKERRQ(ierr);
    
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
    ierr = VecAssemblyBegin(obj->rhs); //CHKERRQ(ierr);
    ierr = VecAssemblyEnd(obj->rhs); //CHKERRQ(ierr);
    
    ierr = VecAssemblyBegin(obj->current_temperature_field_local); //CHKERRQ(ierr);
    ierr = VecAssemblyEnd(obj->current_temperature_field_local); //CHKERRQ(ierr);
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
    ierr = VecDestroy(obj->rhs); //CHKERRQ(ierr);
    ierr = VecDestroy(obj->sol); //CHKERRQ(ierr);
    ierr = MatDestroy(obj->Amat); //CHKERRQ(ierr);
    ierr = KSPDestroy(obj->ksp); //CHKERRQ(ierr);
    

    ierr = VecDestroy(obj->current_temperature_field_local);
    ierr = MatDestroy(obj->stiffness_matrix); //CHKERRQ(ierr);

    return;
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Function to View the matrix and vectors that have been created in an m-file
 %% Note: Assumes all final assemblies of matrices and vectors have been performed
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Petsc_View(PETSC_STRUCT obj, PetscViewer viewer)
{
    PetscErrorCode ierr;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "results.m", &viewer); //CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB); //CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)obj.Amat,"Amat"); //CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)obj.rhs,"rhs"); //CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)obj.sol,"sol"); //CHKERRQ(ierr);
    ierr = MatView(obj.Amat,viewer); //CHKERRQ(ierr);
    ierr = VecView(obj.rhs, viewer); //CHKERRQ(ierr);
    ierr = VecView(obj.sol, viewer); //CHKERRQ(ierr);
    return;
}
