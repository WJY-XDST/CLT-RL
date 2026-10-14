#ifndef __LK_MOTOR_H__
#define __LK_MOTOR_H__

#include "stm32h7xx_hal.h"
#include "can_base.h"
#include "linux_list.h" 

typedef struct
{
	uint16_t id;
	uint16_t online_cnt;
	float torque_current;
  uint8_t temperature;
	uint8_t error;
	float angle;
	float torque;
	int16_t speed;
	float speed_rpm;
  float ecd;
  float Rawecd;
  float omega;
  float ecd_offset;
	float out_torque;
	int16_t output;
	CAN_TxBuffer txBufferInSpeed;
	CAN_TxBuffer txBufferforData;
	CAN_TxBuffer txBufferforPositionSpeed;
	CAN_TxBuffer txBufferforSpeed;
	CAN_TxBuffer txBufferforCurrent;
	CAN_TxBuffer txBufferInTortque;
	float kp;
	float kd;

}LK_Motor;

extern CAN_TxBuffer txBufferInLKGB;
extern LK_Motor lkmotor[4];

void LK_Motor_TorqueControl(LK_Motor* motor);//单电机
void LK_MotorInit(LK_Motor* motor, uint16_t ecd_Offset,CanNumber canx, uint16_t id);
void LK_CanUpdata(CAN_Object canx,CAN_RxBuffer can);
void LK_MotorData_Task(void);
void LK_Motor_CanOutput(LK_Motor* motor);
void LK_Motor_GBOutput(LK_Motor* motor);//多电机
void LK_Motor_GBCanOutput(void);//多电机
void LK_MotorFillData(LK_Motor* motor,float torque);
void LK_MotorFillDataInSpeed(LK_Motor* motor,float speed);//单电机
void LK_Motor_CanOutputInSpeed(LK_Motor* motor);

#endif 
