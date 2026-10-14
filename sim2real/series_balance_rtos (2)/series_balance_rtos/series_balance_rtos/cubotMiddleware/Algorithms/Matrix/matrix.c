/**
 ******************************************************************************
 * @file    matrix.c/h
 * @brief   Matrix/vector calculation. 矩阵/向量运算(二次封装)
 * @author  
 ******************************************************************************
 * Copyright (c) 2023 Team CUBOT
 * All rights reserved.
 ******************************************************************************
 */
#include "stm32h750xx.h"
#include "arm_math.h"
/* Initialise Matrix Instance matrix with Rows, Columns and data array(standard_matrix) */
/**
  * @brief矩阵初始化
  */
arm_matrix_instance_f32 matrix_init(arm_matrix_instance_f32* matrix_out,int Rows,int Columns,float32_t  *standard_matrix )
{	

	arm_mat_init_f32(matrix_out, Rows, Columns,standard_matrix );	
	return *matrix_out;
		
}

/* calculation of matrix_in transpose */
/**
  * @brief//求转置矩阵
  */

//arm_matrix_instance_f32 matrix_tansport(arm_matrix_instance_f32* matrix_in,arm_matrix_instance_f32* matrix_out)
//{
//	arm_mat_trans_f32(matrix_in, matrix_out);
//	return *matrix_out;
//	
//}
arm_matrix_instance_f32 matrix_tansport(arm_matrix_instance_f32* matrix_in)
{
	arm_matrix_instance_f32 matrix_out;
	arm_mat_trans_f32(matrix_in, &matrix_out);
	return matrix_out;
	
}
/* calculation of matrix_A Multiply with matrix_B */
/**
  * @brief//求矩阵A*矩阵B
  */

arm_matrix_instance_f32 matrix_multy(arm_matrix_instance_f32* matrix_A,arm_matrix_instance_f32* matrix_B,arm_matrix_instance_f32* matrix_out)
{
	arm_mat_mult_f32(matrix_A,matrix_B, matrix_out);
	return *matrix_out;
}

/* calculation of Inverse((Transpose(matrix) * matrix) */
/**
  * @brief//求逆矩阵
  */
arm_matrix_instance_f32 matrix_inversed(arm_matrix_instance_f32* matrix,arm_matrix_instance_f32* matrix_out)
{
arm_mat_inverse_f32(matrix,matrix_out);
	return *matrix_out;
	
}

//用前边求出的转置矩阵求
arm_matrix_instance_f32  matrix_inverse_already(arm_matrix_instance_f32* matrix_multy,arm_matrix_instance_f32* matrix_transeport,arm_matrix_instance_f32* matrix_inverse,arm_matrix_instance_f32* matrix_out)
{
	
	arm_mat_inverse_f32(matrix_multy, matrix_inverse);
	arm_mat_mult_f32(matrix_inverse, matrix_transeport,matrix_out);
}

/* calculation of (Inverse((Transpose(matrix) * matrix)) *  Transpose(matrix)) */
/**
  * @brief//逆矩阵*转置矩阵
  */

arm_matrix_instance_f32  matrix_inverse_multy_teansport
(arm_matrix_instance_f32* inverse_matrix,arm_matrix_instance_f32*transport_matrix,arm_matrix_instance_f32* matrix_out)
{
	
	arm_mat_mult_f32(inverse_matrix, transport_matrix, matrix_out);
	return *matrix_out;
}


	
