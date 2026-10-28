#include "pid.h"
#include "et08.h"
#include "filter.h"
#include "WBR.h"
#include "additional_con.h"
/**
  * @brief  限幅
  */
float AmplitudeLimit(float input,float amplitude)
{
	if(input<-amplitude)
		return -amplitude;
	else if(input>amplitude)
		return amplitude;
	else return input;
}


/**
  * @brief 单环PID初始化
  */
void BasePID_Init(BasePID_Object* base_pid, float kp, float ki, float kd, float detach)
{
	base_pid->KiPartDetachment = detach;
	
	base_pid->Kp = kp;
	base_pid->Ki = ki;
	base_pid->Kd = kd;

	base_pid->KpPart = 0;
	base_pid->KiPart = 0;
	base_pid->KdPart = 0;
	
}


/**
  * @brief 单环比例积分速度控制
  */
float BasePID_SpeedControl(BasePID_Object* base_pid, float target_speed, float feedback_speed)
{
	base_pid->Error = target_speed - feedback_speed;
	
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KiPart += base_pid->Error * base_pid->Ki;
	
	if(base_pid->Error > base_pid->KiPartDetachment)
	{
		base_pid->KiPart = 0;
	}
	else if(base_pid->Error < -(base_pid->KiPartDetachment))
	{
		base_pid->KiPart = 0;
	}
	
	base_pid->Out = base_pid->KpPart + base_pid->KiPart;
	
	return base_pid->Out;
}

int32_t BasePID_AngleControl(BasePID_Object* base_pid, float target_angle, float feedback_angle, float feedback_angle_speed)
{
	base_pid->Error = target_angle - feedback_angle;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KiPart += base_pid->Error * base_pid->Ki;
//	base_pid->KiPartDetachment = 10;
//	if(base_pid->Error > base_pid->KiPartDetachment)
//	{
//		base_pid->KiPart = 0;
//	}
//	else if(base_pid->Error < -(base_pid->KiPartDetachment))
//	{
//		base_pid->KiPart = 0;
//	}
	if(base_pid->KiPart > 20000)
		base_pid->KiPart = 20000;
	base_pid->KdPart = (-1) * base_pid->Kd * feedback_angle_speed;
	base_pid->Out = base_pid->KpPart + base_pid->KiPart + base_pid->KdPart;
	
	return base_pid->Out;
}


int32_t BasePID_FollowControl(BasePID_Object* base_pid, float target_angle, float feedback_angle, float feedback_angle_speed)
{
	base_pid->Error = target_angle - feedback_angle;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KiPart += base_pid->Error * base_pid->Ki;
	base_pid->KiPartDetachment = 1000;
	if(base_pid->Error > base_pid->KiPartDetachment)
	{
		base_pid->KiPart = 0;
	}
	else if(base_pid->Error < -(base_pid->KiPartDetachment))
	{
		base_pid->KiPart = 0;
	}
	
	base_pid->KdPart = (base_pid->Error-base_pid->LastError)*base_pid->Kd;
	base_pid->LastError=base_pid->Error;
	base_pid->Out = base_pid->KpPart + base_pid->KiPart + base_pid->KdPart;
	if(base_pid->Out>=80)
		base_pid->Out=80;
	if(base_pid->Out<=-80)
		base_pid->Out=-80;
	return base_pid->Out;
}

/**
  * @brief 缓冲能量剩余目标值为20时，功率pid控制
  */
float BasePID_BaseControl(BasePID_Object* base_pid, float Chassis_Power_Buffer, float Chassis_Power)
	{
	if(Chassis_Power_Buffer>=20)
	{
	float Target_Power_Buff=20;       //缓冲能量剩余目标值为20
  base_pid->Error = -Target_Power_Buff+Chassis_Power_Buffer;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KiPart += base_pid->Error * base_pid->Ki;
	if(base_pid->Error > base_pid->KiPartDetachment)
	{
		base_pid->KiPart = 0;
	}
	else if(base_pid->Error < -(base_pid->KiPartDetachment))
	{
		base_pid->KiPart = 0;
	}
	base_pid->Out = base_pid->KpPart + base_pid->KiPart + base_pid->KdPart;
	return base_pid->Out;
  }
	else
	return 0;
}


/**
  * @brief 缓冲能量剩余目标值为10时，功率pid控制
  */
float BasePID_PowerControl(BasePID_Object* base_pid, float Chassis_Power_Buffer, float Chassis_Power)
	{
	float Target_Power_Buffer=10;    //缓冲能量剩余目标值为10
  base_pid->Error = -Target_Power_Buffer+Chassis_Power_Buffer;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KiPart += base_pid->Error * base_pid->Ki;
	if(base_pid->Error > base_pid->KiPartDetachment)
	{
		base_pid->KiPart = 0;
	}
	else if(base_pid->Error < -(base_pid->KiPartDetachment))
	{
		base_pid->KiPart = 0;
	}
	base_pid->KdPart = (-1) * base_pid->Kd * Chassis_Power ;
	base_pid->Out = base_pid->KpPart + base_pid->KiPart + base_pid->KdPart;
	return base_pid->Out;
}

float TARget_length_l,TARget_length_r;
float BasePID_lengthControl_l(BasePID_Object* base_pid, float target_length,float roll_com, float length)
{
	if(TARget_length_l<target_length)
		TARget_length_l+=0.0004;
	if(TARget_length_l>target_length)
		TARget_length_l-=0.0004;
	base_pid->Error = TARget_length_l-roll_com- length;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KdPart = (base_pid->Error-base_pid->LastError) * base_pid->Kd;
	base_pid->KiPart += base_pid->Error * base_pid->Ki;
	if(base_pid->Error > base_pid->KiPartDetachment)
	{
		base_pid->KiPart = 0;
	}
	else if(base_pid->Error < -(base_pid->KiPartDetachment))
	{
		base_pid->KiPart = 0;
	}
	base_pid->Out = (base_pid->KpPart + base_pid->KdPart + base_pid->KiPart);
	base_pid->LastError=base_pid->Error;
	return base_pid->Out;
}

float BasePID_lengthControl_r(BasePID_Object* base_pid, float target_length,float roll_com, float length)
{
	if(TARget_length_r<target_length)
		TARget_length_r+=0.0004;
	if(TARget_length_r>target_length)
		TARget_length_r-=0.0004;
	base_pid->Error = TARget_length_r+roll_com- length;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KdPart = (base_pid->Error-base_pid->LastError) * base_pid->Kd;

		base_pid->KiPart += base_pid->Error * base_pid->Ki;
	if(base_pid->Error > base_pid->KiPartDetachment)
	{
		base_pid->KiPart = 0;
	}
	else if(base_pid->Error < -(base_pid->KiPartDetachment))
	{
		base_pid->KiPart = 0;
	}

	base_pid->Out = (base_pid->KpPart + base_pid->KdPart + base_pid->KiPart);
	base_pid->LastError=base_pid->Error;
	return base_pid->Out;
}
/*转向控制测试*/
float BasePID_Turncontrol(BasePID_Object* base_pid, float taget_yaw, float yaw,float yaw_speed)
{
	base_pid->Error = taget_yaw - yaw;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	base_pid->KdPart = -yaw_speed * base_pid->Kd;
	
	base_pid->Out = base_pid->KpPart + base_pid->KdPart;
	return base_pid->Out;
}
/**
  * @brief 单环比例积分roll控制   
  */
float BasePID_rollControl(BasePID_Object* base_pid, float target_length, float length)
{
	base_pid->Error = target_length - length;
	base_pid->KpPart = base_pid->Error * base_pid->Kp;
	if(fabs(base_pid->Error) < 0.5)
		base_pid->Error = 0;
	if(lqr_ready_flag==1)
	{
		base_pid->KiPart += base_pid->Error * base_pid->Ki;
	}
	else 
	{
		base_pid->KiPart =0;
	}
	if(autojump_double_flag==1||nextjump_flag!=1||groundoff_flag==1)
		base_pid->KiPart = 0;
	
	if(base_pid->KiPart > 0.1)
		base_pid->KiPart = 0.1;
	if(base_pid->KiPart < -0.1)
		base_pid->KiPart = -0.1;
	
	
	
	base_pid->Out = base_pid->KpPart+base_pid->KiPart;
	
	if(base_pid->Out > 0.2)
		base_pid->Out = 0.2;
	if(base_pid->Out < -0.2)
		base_pid->Out = -0.2;
	
	return base_pid->Out;
}