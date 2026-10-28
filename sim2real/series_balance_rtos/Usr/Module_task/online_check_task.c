/**
  *********************************************************************
  * @file      online_check_task.c/h
  * @brief     用于判断各个模块离线
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************/
#include "DMmotor.h"
#include "WBR.h"
int left_leg_cnt=0;
int right_leg_cnt=0;
int leg_online=0;
volatile uint8_t leg_reset_flag=0;//电机上电后需要复位标志位

uint8_t online_check()
{
//	rc_Ctrl.onlineCheckCnt++;
//	if(rc_Ctrl.onlineCheckCnt > 50)
//		rc_Ctrl.isOnline = 0;
//	else
//	{
//		rc_Ctrl.isOnline = 1;//该判断已移至板间通信中
//	}

//	return rc_Ctrl.isOnline;
}

uint8_t DM_online_check()
{
	left_leg_cnt ++;
	right_leg_cnt ++;
	if(left_leg_cnt > 100 && right_leg_cnt >100)
		leg_online = 0;
	else 
		leg_online = 1;
	if(leg_online == 0)
		leg_reset_flag=1;//电机上电后需要复位标志位
}
int wheel_left,wheel_right;
int8_t wheel_left_online = 1;
int8_t wheel_right_online = 1;
void wheel_online_check()
{
	wheel_left++;
	wheel_right++;
	if(wheel_left < 250)
		wheel_left_online = 1;
	else 
		wheel_left_online = 0;
	if(wheel_right < 250)
		wheel_right_online = 1;
	else 
		wheel_right_online = 0;
		
}
int online_check_count;
void online_check_task(void *param)
{
	TickType_t xLastWakeTime;
	xLastWakeTime = xTaskGetTickCount();
	online_check_count = 0;
	while(1)
	{
		online_check();
//		DM_online_check();
		wheel_online_check();
		vTaskDelayUntil(&xLastWakeTime,2);//2ms
	}
}
  