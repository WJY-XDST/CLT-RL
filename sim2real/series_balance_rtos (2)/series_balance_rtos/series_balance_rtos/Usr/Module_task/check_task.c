/**
  *********************************************************************
  * @file      additional_task.c/h
  * @brief     用于处理处理底盘其他任务，如滑铲，离地，跳跃等
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */
#include "motor.h"
#include "can_base.h"
#include "all_init.h"
#include "driver_counter.h"
#include "torque.h"
#include "A1_stm32.h"
#include "WBR.h"
#include "wheel.h"
#include "torque.h"
#include "ins.h"
#include "HI04.h"
#include "additional_con.h"
/*freertos*/
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
int addition_count;
UBaseType_t uxHighWaterMark_addition;
/*外加任务*/
void addition_task(void *param)
{
	TickType_t xLastWakeTime;
	xLastWakeTime = xTaskGetTickCount();
	addition_count = 0;
	while(1)
	{
		addition_count++;
		ground_dection(&VMC_joint_left,&VMC_joint_right);//离地检测
auto_jump();
//		gleiten_dection(&VMC_joint_left,&VMC_joint_right);//滑铲检测
		//collide_check();//横向碰撞检测
//		spin_control();//自旋
		/*speed_perior*/
//		Wheel_predictor_com(&Wheel,&PErior);
		/*supercap_interaction*/
//		Supercap_inter(&can1);
		/*joint_motor_check*/
//		A1_motor_check();
//		type_symbol();
		uxHighWaterMark_addition = uxTaskGetStackHighWaterMark(NULL );
		vTaskDelayUntil(&xLastWakeTime,2);//2ms
	
	}

}