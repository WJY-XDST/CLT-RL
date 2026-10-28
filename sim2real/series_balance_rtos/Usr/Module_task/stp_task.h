#ifndef STP_TASK_H
#define STP_TASK_H
#include "all_init.h"
#include "uart_base.h"

#define ANGLE_PER_FRAME 12
#define HEADER 0x54
#define POINT_PER_PACK 12
#define VERLEN  0x2C 	//低五位是一帧数据接收到的点数，目前固定是12，高三位固定为1
#define STP_rxBufferLengh  9



typedef struct __attribute__((packed))
{
	uint16_t temperature;
  uint16_t distance;//距离
	uint16_t intensity;//置信度
}RANGEFrameTypeDef;

extern RANGEFrameTypeDef Pack_Data;//接收的数据储存在这个变量之中

extern UART_RxBuffer uart7_buffer;
extern	uint32_t stp_distance;
uint8_t stp_callback(uint8_t * recBuffer, uint16_t len);
void STP_task(void *param);
void stp_modify(uint8_t * recBuffer);
void data_process(void);


#endif
