#ifndef __PID_H_
#define __PID_H_

#include "stm32h7xx_hal.h"



/**
  * @brief  单环PID 
  */
typedef struct 
{	
	float Kp, Ki, Kd;
	float Error;
	float KpPart, KiPart, KdPart;
	float Out;
	float KiPartDetachment;
	float LastError;
}BasePID_Object;


/**
  * @brief  双环PID
  */
typedef struct 
{
	BasePID_Object* ShellPID;
	BasePID_Object* CorePID;
}DualPID_Object;
 

/**
  * @brief  		限幅
  * @param[in]	input		输入量
  * @param[in]	output	输出量
  */
float AmplitudeLimit(float input,float amplitude);


/**
  * @brief 单环PID初始化
  */
void BasePID_Init(BasePID_Object* base_pid, float kp, float ki, float kd, float detach);


/**
  * @brief 双环PID初始化
  */
void DualPID_Init(DualPID_Object* dual_pid, float shell_kp, float shell_ki, float shell_kd, float core_kp, float core_ki, float core_kd, float detach);



/**
  * @brief 单环比例积分速度控制
  */
float BasePID_SpeedControl(BasePID_Object* base_pid, float target_speed, float feedback_speed);



/**
  * @brief 单环比例积分速度控制, 角速度不依靠IMU数据，仅靠编码器进行角度控制
  */
int32_t BasePID_AngleControlWithoutIMU(BasePID_Object* base_pid, float target_angle, float feedback_angle, float feedback_angle_speed);


/**
  * @brief 单环比例微分角度控制, 角速度不依靠IMU数据，仅靠编码器进行角度控制
  */
int32_t BasePID_AngleControl(BasePID_Object* base_pid, float target_angle, float feedback_angle, float feedback_angle_speed);
float BasePID_BaseControl(BasePID_Object* base_pid, float Chassis_Power_Buffer, float Chassis_Power);
int32_t BasePID_HoldControl(BasePID_Object* base_pid, float target_angle, float feedback_angle, float feedback_angle_speed);
float BasePID_PowerControl(BasePID_Object* base_pid, float Chassis_Power_Buffer, float Chassis_Power);


float BasePID_lengthControl_l(BasePID_Object* base_pid, float target_length, float roll_com,float length);
float BasePID_lengthControl_r(BasePID_Object* base_pid, float target_length, float roll_com,float length);

float BasePID_rollControl(BasePID_Object* base_pid, float target_length, float length);
int32_t BasePID_FollowControl(BasePID_Object* base_pid, float target_angle, float feedback_angle, float feedback_angle_speed);
float BasePID_Turncontrol(BasePID_Object* base_pid, float taget_yaw, float yaw,float yaw_speed);
float BasePID_rollControl(BasePID_Object* base_pid, float target_length, float length);

#endif

