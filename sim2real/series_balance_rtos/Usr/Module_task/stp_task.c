/**
  *********************************************************************
  * @file      stp_task.c/h
  * @brief     该任务用于stp-23模块测距
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */
#include "usart.h"
#include "stp_task.h"
#include "uart_base.h"

uint8_t STP_recData[STP_rxBufferLengh]__attribute__((at(0x24007800)));


/**
	* @brief  创建串口缓存区数据结构
	*/
UART_RxBuffer uart7_buffer={
	.Data = STP_recData,
	.Size = STP_rxBufferLengh
};
UBaseType_t uxHighWaterMark_stp;
uint32_t stp_distance;
int stp_count;//测试执行频率	
void STP_task(void *param) 
{
	uint8_t * RecBuffer;
	stp_count = 0;
	while(1)
	{
		stp_count ++;
		xQueueReceive(uart7.g_xQueueUart,&RecBuffer,portMAX_DELAY);
		stp_modify(RecBuffer);
		uxHighWaterMark_stp = uxTaskGetStackHighWaterMark(NULL);
	}
			
}

RANGEFrameTypeDef Pack_Data;//接收的数据储存在这个变量之中
	int8_t Checksum;
void stp_modify(uint8_t * recBuffer)
{

	if(recBuffer[0] == 0x59 && recBuffer[0] == 0x59)
	{
			for(int i = 0;i < 8;i++)
		{
			Checksum += recBuffer[i];
		}
			if(Checksum == recBuffer[8])
		{
			Pack_Data.intensity = recBuffer[4]|recBuffer[5]<<8;
			Pack_Data.temperature = (recBuffer[6]|recBuffer[7]<<8) / 8 - 256;
				if(Pack_Data.intensity > 60)
					Pack_Data.distance = recBuffer[2]|recBuffer[3]<<8;		
				else Pack_Data.distance = 5000;
		}
		Checksum = 0;
	}
		
}
	


uint8_t stp_callback(uint8_t * recBuffer, uint16_t len)
{
	xQueueSendFromISR(uart7.g_xQueueUart, &(recBuffer), 0);
	return 0;
}

