/**
  *********************************************************************
  * @file      ins_task.c/h
  * @brief     该任务是用mahony方法获取机体姿态，同时获取机体在绝对坐标系下的运动加速度
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */
	
#include "INS_task.h"
#include "bmi088.h"
#include "quaternions_EKF.h"
#include "FreeRTOS.h"
#include "task.h"
#include "mahony_filter.h"

INS_t1 INS;
IMU_InitData_t BMI088;
struct MAHONY_FILTER_t mahony;
Axis3f Gyro,Accel;
float gravity[3] = {0, 0, 9.81f};

uint32_t INS_DWT_Count = 0;
float ins_dt = 0.0f;
float ins_time;
int stop_time;

void INS_Init1(void)
{ 
	 mahony_init(&mahony,1.0f,0.0f,0.001f);
   INS.AccelLPF = 0.0089f;
}

void INS_task(void)
{
	TickType_t xLastWakeTime;
	xLastWakeTime = xTaskGetTickCount();
	 INS_Init1();
	 while(1)
	 {  
		ins_dt = DWT_GetDeltaT(&INS_DWT_Count);
    
		mahony.dt = ins_dt;

		BMI088_Read(&bmi088.bmi088_Data);

		INS.Accel[X] = bmi088.bmi088_Data.accel[X];
		INS.Accel[Y] = bmi088.bmi088_Data.accel[Y];
		INS.Accel[Z] = bmi088.bmi088_Data.accel[Z];
		INS.Gyro[X] = bmi088.bmi088_Data.gyro[X];
		INS.Gyro[Y] = bmi088.bmi088_Data.gyro[Y];
		INS.Gyro[Z] = bmi088.bmi088_Data.gyro[Z];
		
		Param_Correction(-1,INS.Gyro,INS.Accel);
		 
		 Gyro.x=INS.Gyro[0];
		Gyro.y=INS.Gyro[1];
		Gyro.z=INS.Gyro[2];
		 Accel.x=INS.Accel[0];
		Accel.y=INS.Accel[1];
		Accel.z=INS.Accel[2];
		mahony_input(&mahony,Gyro,Accel);
		mahony_update(&mahony);
		mahony_output(&mahony);
	    RotationMatrix_update(&mahony);
				
		INS.q[0]=mahony.q0;
		INS.q[1]=mahony.q1;
		INS.q[2]=mahony.q2;
		INS.q[3]=mahony.q3;
       
      // 将重力从导航坐标系n转换到机体系b,随后根据加速度计数据计算运动加速度
		float gravity_b[3];
    EarthFrameToBodyFrame(gravity, gravity_b, INS.q);
    for (uint8_t i = 0; i < 3; i++) // 同样过一个低通滤波
    {
      INS.MotionAccel_b[i] = (INS.Accel[i] - gravity_b[i]) * ins_dt / (INS.AccelLPF + ins_dt) 
														+ INS.MotionAccel_b[i] * INS.AccelLPF / (INS.AccelLPF + ins_dt); 
//			INS.MotionAccel_b[i] = (INS.Accel[i] ) * dt / (INS.AccelLPF + dt) 
//														+ INS.MotionAccel_b[i] * INS.AccelLPF / (INS.AccelLPF + dt);			
		}
		BodyFrameToEarthFrame(INS.MotionAccel_b, INS.MotionAccel_n, INS.q); // 转换回导航系n
		
		//死区处理
		if(fabsf(INS.MotionAccel_n[0])<0.02f)
		{
		  INS.MotionAccel_n[0]=0.0f;	//x轴
		}
		if(fabsf(INS.MotionAccel_n[1])<0.02f)
		{
		  INS.MotionAccel_n[1]=0.0f;	//y轴
		}
		if(fabsf(INS.MotionAccel_n[2])<0.04f)
		{
		  INS.MotionAccel_n[2]=0.0f;//z轴
			stop_time++;
		}
		if(stop_time>10)
		{//静止10ms
		  stop_time=0;
			INS.v_n=0.0f;
		}
    		
		if(ins_time>3000.0f)
		{
			INS.v_n=INS.v_n+INS.MotionAccel_n[1]*0.001f;
		  INS.x_n=INS.x_n+INS.v_n*0.001f;
			INS.ins_flag=1;//四元数基本收敛，加速度也基本收敛，可以开始底盘任务
			// 获取最终数据
          INS.Pitch=mahony.roll;
		  INS.Roll=mahony.pitch ;
		  INS.Yaw=mahony.yaw;
		
		//INS.YawTotalAngle=INS.YawTotalAngle+INS.Gyro[2]*0.001f;
			
			if (INS.Yaw - INS.YawAngleLast > 3.1415926f)
			{
					INS.YawRoundCount--;
			}
			else if (INS.Yaw - INS.YawAngleLast < -3.1415926f)
			{
					INS.YawRoundCount++;
			}
			INS.YawTotalAngle = 6.283f* INS.YawRoundCount + INS.Yaw;
			INS.YawAngleLast = INS.Yaw;
		}
		else
		{
		 ins_time++;
		}
		
    vTaskDelayUntil(&xLastWakeTime,1);
	}
} 

uint8_t correction_flag=1;
static void Param_Correction(float roll_com, float gyro[3], float accel[3])
{
    static float lastYawOffset, lastPitchOffset, lastRollOffset;
    static float c_11, c_12, c_13, c_21, c_22, c_23, c_31, c_32, c_33;
    float cosPitch, cosYaw, cosRoll, sinPitch, sinYaw, sinRoll;

    if (correction_flag == 1) {
        cosYaw   = arm_cos_f32(0);
        cosPitch = arm_cos_f32(0);
        cosRoll  = arm_cos_f32(roll_com/ 57.295779513f);
        sinYaw   = arm_sin_f32(0);
        sinPitch = arm_sin_f32(0);
        sinRoll  = arm_sin_f32(roll_com/ 57.295779513f);

        // 1.yaw(alpha) 2.pitch(beta) 3.roll(gamma)
        c_11        = cosYaw * cosRoll + sinYaw * sinPitch * sinRoll;
        c_12        = cosPitch * sinYaw;
        c_13        = cosYaw * sinRoll - cosRoll * sinYaw * sinPitch;
        c_21        = cosYaw * sinPitch * sinRoll - cosRoll * sinYaw;
        c_22        = cosYaw * cosPitch;
        c_23        = -sinYaw * sinRoll - cosYaw * cosRoll * sinPitch;
        c_31        = -cosPitch * sinRoll;
        c_32        = sinPitch;
        c_33        = cosPitch * cosRoll;
        correction_flag = 0;
    }
    float gyro_temp[3];
    for (uint8_t i = 0; i < 3; ++i)
        gyro_temp[i] = gyro[i] *1;

    gyro[X] = c_11 * gyro_temp[X] +
              c_12 * gyro_temp[Y] +
              c_13 * gyro_temp[Z];
    gyro[Y] = c_21 * gyro_temp[X] +
              c_22 * gyro_temp[Y] +
              c_23 * gyro_temp[Z];
    gyro[Z] = c_31 * gyro_temp[X] +
              c_32 * gyro_temp[Y] +
              c_33 * gyro_temp[Z];

    float accel_temp[3];
    for (uint8_t i = 0; i < 3; ++i)
        accel_temp[i] = accel[i];

    accel[X] = c_11 * accel_temp[X] +
               c_12 * accel_temp[Y] +
               c_13 * accel_temp[Z];
    accel[Y] = c_21 * accel_temp[X] +
               c_22 * accel_temp[Y] +
               c_23 * accel_temp[Z];
    accel[Z] = c_31 * accel_temp[X] +
               c_32 * accel_temp[Y] +
               c_33 * accel_temp[Z];
			   
//	Gyro.x=gyro[X];
//	Gyro.y=gyro[Y];
//	Gyro.z=gyro[Z];
//	
//	Accel.x= accel[X];
//	Accel.y= accel[X];
//	Accel.z= accel[X];

}



