#ifndef _WHEEL_H
#define _WHEEL_H
#include "stm32h7xx_hal.h"
#include "pid.h"
#include "motor.h"
#include "et08.h"
#include "uart_base.h"

struct wheel_Motor_Info
{
    float   Target_Angle;
	float   Last_Can_Angle;
	float   Angle_speed;
    float	angle;
	float	angle_last;
};

typedef struct
{
	
 
	struct wheel_Motor_Info S_joint[2];
	struct 
	{ 
		
		Motor motor[2];						//< 姿态电机结构													
		BasePID_Object turnPID[2];		//< 转向角度控制结构体
	}Motors3508;
	float S_L;
	float S_L_last;
	float S_L_dot;
	float S_R;
	float S_R_last;
	float S_R_dot;
	float S;
	float S_last;
	float S_dot;
	float S_target;
	float TW_adapt_L;
	float TW_adapt_R;//轮力矩预测估计补偿
}wheel;


void wheel_joint_Init(wheel* Wheel, CanNumber canx);


extern wheel Wheel;
extern int8_t chassis_reset_flag;//底盘复位标志

void Angular_accum(wheel* Wheel);//角度累加函数
void S_count(wheel* Wheel,RC_Ctrl* rc_ctrl);
void zerotorque_3508();

void benchchassis_conrol(wheel* Wheel,RC_Ctrl* rc_ctrl);


#endif