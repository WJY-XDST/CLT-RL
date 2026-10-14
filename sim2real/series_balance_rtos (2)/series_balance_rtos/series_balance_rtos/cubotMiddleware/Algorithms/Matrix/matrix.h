/**
 ******************************************************************************
 * @file    matrix.c/h
 * @brief   Matrix/vector calculation. æÿ’Û/œÚ¡ø‘ÀÀ„
 * @author  Nico ly
 ******************************************************************************
 * Copyright (c) 2023 Team CUBOT
 * All rights reserved.
 ******************************************************************************
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "arm_math.h"
#define pi 3.14159
arm_matrix_instance_f32 matrix_init(arm_matrix_instance_f32* matrix_out,int Rows,int Columns, float32_t* standard_matrix);
arm_matrix_instance_f32 matrix_tansport(arm_matrix_instance_f32* matrix_in);
arm_matrix_instance_f32 matrix_multy(arm_matrix_instance_f32* matrix_A,arm_matrix_instance_f32* matrix_B,arm_matrix_instance_f32* matrix_out);
arm_matrix_instance_f32  matrix_inverse(float32_t* matrix,int Rows,int Columns,arm_matrix_instance_f32* matrix_out);
arm_matrix_instance_f32  matrix_inverse_already(arm_matrix_instance_f32* matrix,arm_matrix_instance_f32* matrix_out);
arm_matrix_instance_f32  matrix_inverse_multy_teansport
(arm_matrix_instance_f32* inverse_matrix,arm_matrix_instance_f32*transport_matrix,arm_matrix_instance_f32* matrix_out);
arm_matrix_instance_f32  matrix_in_tran_tran(arm_matrix_instance_f32* matrix_inverse_multy_teansport,arm_matrix_instance_f32*matrix_B,arm_matrix_instance_f32* matrix_out);
void matrix_trans(arm_matrix_instance_f32* matrix,float32_t *trans_matrix,float32_t T[16]);
void solve_endpose(arm_matrix_instance_f32* matrix,float32_t *trans_matrix);













#endif  // MATRIX_H
