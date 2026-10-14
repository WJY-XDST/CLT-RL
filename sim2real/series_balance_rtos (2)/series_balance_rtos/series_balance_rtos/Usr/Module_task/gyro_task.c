/**
  *********************************************************************
  * @file      gyro_task.c/h
  * @brief     该任务获取陀螺仪姿态，以及遥控离线判断
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */
	
#include "bmi088.h"
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
#include "et08.h"
#include "HI04.h"
#include "arm_math.h"
int gyro_count;//测试执行频率
void centripetal_force_com(float w);


void Gyrodata_task(void *param)
{
	TickType_t xLastWakeTime;
	xLastWakeTime = xTaskGetTickCount();
	gyro_count = 0;
	while(1)
	{
		INS_attitude = INS_GetAttitude(IMU_data);//陀螺仪数据获取
		gyro_count++;
		//centripetal_force_com(INS_attitude->yaw_speed);
		//vTaskDelay(1);
		vTaskDelayUntil(&xLastWakeTime,2);
	}
}

void HI04data_task(void *param)
{
	
	TickType_t xLastWakeTime;
	xLastWakeTime = xTaskGetTickCount();
	gyro_count = 0;
	uint8_t * RecBuffer;
	while(1)
	{
		//online_check();
		//gyro_count++;
		xQueueReceive(uart8.g_xQueueUart,&RecBuffer,portMAX_DELAY);
		HI04_data_receive(RecBuffer);
		if(gyro_count <= 100)
		HI04_modify(0);
		else 
		HI04_modify(1);
	}


}
float phi = 0.64;//圆心与陀螺仪夹角
/*由于陀螺仪安装位置偏心，故作向心力补偿尝试解决*/
float bmi088_ay,bmi088_ax;//向心力分量补偿
float kx=0,ky=0;
void centripetal_force_com(float w)
{
	float r = 0.1;//圆心与陀螺仪之间距离，陀螺仪圆周半径
	//phi = 0.64;
	float a;
	a  = w*w*r;//向心力
	bmi088_ay = ky*a*arm_sin_f32(phi);//向心力y分量补偿
	bmi088_ax = - kx*a*arm_cos_f32(phi);//向心力x分量补偿
}