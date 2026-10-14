/**@file  driver_iic.c
* @brief    板级支持包，串口管理器配置文件，用户回调重定义
* @details  主要包括构建串口管理器，提供串口初始化和用户回调重定义
* @author      RyanJiao  any question please send mail to 1095981200@qq.com

* @date        2022-2-20
* @version     V1.1
* @copyright    Copyright (c) 2021-2121  中国矿业大学CUBOT战队
**********************************************************************************
* @attention
* 硬件平台: STM32H750VBT \n
* SDK版本：-++++
* @par 修改日志:
* <table>
* <tr><th>Date        <th>Version  <th>Author    <th>Description
* <tr><td>2021-8-12  <td>1.0      <td>RyanJiao  <td>创建初始版本
* </table>
*
**********************************************************************************
*/
#include "driver_iic.h"
#include "i2c.h"


static void I2C2_Error(uint8_t Addr)
{
	/* 恢复I2C寄存器为默认值 */
	HAL_I2C_DeInit(&hi2c2);
	/* 重新初始化I2C外设 */
	HAL_I2C_Init(&hi2c2);
}

static void I2C4_Error(uint8_t Addr)   //
{
	/* 恢复I2C寄存器为默认值 */
	HAL_I2C_DeInit(&hi2c4);
	/* 重新初始化I2C外设 */
	HAL_I2C_Init(&hi2c4);
}


/**
  * @brief  写寄存器，这是提供给上层的接口
  */
int Sensors_I2C_WriteRegister(unsigned char slave_addr,
	unsigned char reg_addr,
	unsigned short len,
	unsigned char* data_ptr)
{
	HAL_StatusTypeDef status = HAL_OK;
	status = HAL_I2C_Mem_Write(&hi2c2, slave_addr, reg_addr, I2C_MEMADD_SIZE_8BIT, data_ptr, len, I2Cx_FLAG_TIMEOUT);
	
	/* 检查通讯状态 */
	if (status != HAL_OK)
	{
		/* 总线出错处理 */
		I2C2_Error(slave_addr);
	  
	}
	while (HAL_I2C_GetState(&hi2c2) != HAL_I2C_STATE_READY)
	{

	}
	/* 检查SENSOR是否就绪进行下一次读写操作 */
	while (HAL_I2C_IsDeviceReady(&hi2c2, slave_addr, I2Cx_FLAG_TIMEOUT, I2Cx_FLAG_TIMEOUT) == HAL_TIMEOUT);
	
	/* 等待传输结束 */
	while (HAL_I2C_GetState(&hi2c2) != HAL_I2C_STATE_READY)
	{

	}
	
}


/**
  * @brief  写寄存器，这是提供给上层的接口
  */
int Sensors_I2C_WriteRegister2(unsigned char slave_addr,
	unsigned char reg_addr,
	unsigned short len,
	unsigned char* data_ptr)
{
	HAL_StatusTypeDef status = HAL_OK;
	status = HAL_I2C_Mem_Write(&hi2c4, slave_addr, reg_addr, I2C_MEMADD_SIZE_8BIT, data_ptr, len, I2Cx_FLAG_TIMEOUT);
	
	/* 检查通讯状态 */
	if (status != HAL_OK)
	{
		/* 总线出错处理 */
		I2C4_Error(slave_addr);
	  
	}
	while (HAL_I2C_GetState(&hi2c4) != HAL_I2C_STATE_READY)
	{

	}
	/* 检查SENSOR是否就绪进行下一次读写操作 */
	while (HAL_I2C_IsDeviceReady(&hi2c4, slave_addr, I2Cx_FLAG_TIMEOUT, I2Cx_FLAG_TIMEOUT) == HAL_TIMEOUT);
	
	/* 等待传输结束 */
	while (HAL_I2C_GetState(&hi2c4) != HAL_I2C_STATE_READY)
	{

	}
	
}

/**
  * @brief  读寄存器，这是提供给上层的接口
  */
int Sensors_I2C_ReadRegister(unsigned char slave_addr,
	unsigned char reg_addr,
	unsigned short len,
	unsigned char* data_ptr)
{
	HAL_StatusTypeDef status = HAL_OK;
	status = HAL_I2C_Mem_Read(&hi2c2, slave_addr, reg_addr, I2C_MEMADD_SIZE_8BIT, data_ptr, len, I2Cx_FLAG_TIMEOUT);
	/* 检查通讯状态 */
	if (status != HAL_OK)
	{
		/* 总线出错处理 */
		I2C2_Error(slave_addr);
	}
	while (HAL_I2C_GetState(&hi2c2) != HAL_I2C_STATE_READY)
	{

	}
	/* 检查SENSOR是否就绪进行下一次读写操作 */
	while (HAL_I2C_IsDeviceReady(&hi2c2, slave_addr, I2Cx_FLAG_TIMEOUT, I2Cx_FLAG_TIMEOUT) == HAL_TIMEOUT);
	/* 等待传输结束 */
	while (HAL_I2C_GetState(&hi2c2) != HAL_I2C_STATE_READY)
	{

	}
	
	return status;
}

/**
  * @brief  读寄存器，这是提供给上层的接口
  */
int Sensors_I2C_ReadRegister2(unsigned char slave_addr,
	unsigned char reg_addr,
	unsigned short len,
	unsigned char* data_ptr)
{
	HAL_StatusTypeDef status = HAL_OK;
	status = HAL_I2C_Mem_Read(&hi2c4, slave_addr, reg_addr, I2C_MEMADD_SIZE_8BIT, data_ptr, len, I2Cx_FLAG_TIMEOUT);
	/* 检查通讯状态 */
	if (status != HAL_OK)
	{
		/* 总线出错处理 */
		I2C4_Error(slave_addr);
	}
	while (HAL_I2C_GetState(&hi2c4) != HAL_I2C_STATE_READY)
	{

	}
	/* 检查SENSOR是否就绪进行下一次读写操作 */
	while (HAL_I2C_IsDeviceReady(&hi2c4, slave_addr, I2Cx_FLAG_TIMEOUT, I2Cx_FLAG_TIMEOUT) == HAL_TIMEOUT);
	/* 等待传输结束 */
	while (HAL_I2C_GetState(&hi2c4) != HAL_I2C_STATE_READY)
	{

	}
	
	return status;
}
