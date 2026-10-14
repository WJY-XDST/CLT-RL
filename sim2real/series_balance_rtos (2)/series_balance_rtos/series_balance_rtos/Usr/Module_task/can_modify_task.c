/**
  *********************************************************************
  * @file      can_modify_task.c/h
  * @brief     该任务用于处理can接收到的数据
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */

#include "can_base.h"
#include "motor.h"
#include "LK_motor.h"

/**
  * @brief  CAN接收处理函数
  */
  UBaseType_t uxHighWaterMark_can;


void can_data_modify_task(void *param)
{
	  CAN_RxBuffer rxBuffer_can;
    CAN_Object *can = param;
	while(1)
	{
		xQueueReceive(can->g_xQueueCan,&rxBuffer_can,portMAX_DELAY);
		if(can==&can2)
		MotorRxCallback(*can,rxBuffer_can);//motor
		if(can==&can1)
		LK_CanUpdata(can1,rxBuffer_can);

		uxHighWaterMark_can = uxTaskGetStackHighWaterMark(NULL);
	}
}
