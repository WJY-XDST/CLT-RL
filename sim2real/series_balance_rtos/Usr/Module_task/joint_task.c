/**
  *********************************************************************
  * @file      joint_task.c/h
  * @brief     该任务用于给电机发送数据控制
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
#include "online_check_task.h"
#include "resurgence_control.h"
#include "gpio.h"
/*freertos*/
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
#include "additional_con.h"
#include "DMmotor.h"
#include "pid.h"


int  testoutput = 0;
uint16_t leg_reset_judge_count=0;//防误识别


UBaseType_t uxHighWaterMark_joint;
int joint_count;//测试执行频率
extern uint8_t leg_reset_count;
void joint_task(void *param)
{
	joint_count = 0;
	int joint_timeclock=0;
	TickType_t xLastWakeTime;
	xLastWakeTime = xTaskGetTickCount();
	while(1)
	{
		joint_count++;
		joint_timeclock++;
		if(joint_timeclock < 1200)
		{
//    DMotor[0].Data.Output = 0;
//		DMotor[1].Data.Output = 0;
//		DMotor[2].Data.Output = 0;
//		DMotor[3].Data.Output = 0;

		}
		if(joint_timeclock == 1200)
			leg_reset_flag = 0;
		if(joint_timeclock > 1200)
		{
			if(leg_reset_flag ==0)
			{				
        torque_LK(&VMC_joint_left,&VMC_joint_right);
				leg_reset_judge_count = 0;
			}
			else if (leg_reset_flag ==1 )
			{
				leg_reset_judge_count++;
				if(leg_reset_judge_count > 50)
				{	
					leg_reset();
				}
				
			}
			torque_motor3508(&Wheel,VMC_joint_left.TW,VMC_joint_right.TW);
		}

		if(joint_count%2==0)
		{
		  MotorCanOutput(can2, 0x200);
		}	


		//vTaskDelay(1);
		uxHighWaterMark_joint = uxTaskGetStackHighWaterMark(NULL);
		vTaskDelayUntil(&xLastWakeTime,1);
	}

}


