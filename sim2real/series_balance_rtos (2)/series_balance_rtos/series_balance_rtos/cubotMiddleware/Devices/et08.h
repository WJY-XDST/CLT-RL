#ifndef ET08_H
#define ET08_H

#include "stm32h7xx_hal.h"
#include "usart.h"
#include "uart_base.h"

#define ET08_rxBufferLengh 35	 //< et08接收缓存区数据长度
#define ET08_dataLengh 35	 //< et08接收数据长度

extern UART_RxBuffer ET08_Rxbuffer;

typedef struct { 
 struct {       
	 uint16_t ch0;       
	 uint16_t ch1;       
	 uint16_t ch2;      
	 uint16_t ch3;  
     uint8_t  sA; 	 
	 uint8_t  s1; 
	 uint8_t  s2;
     uint8_t  sD;	 
	 uint8_t  s1_last; 
	 uint8_t  s2_last; 
	}rc; 
 	uint8_t  isUnpackaging;  	 //< 解算状态标志位，解算过程中不读取数据
	uint8_t  isOnline;
	uint32_t onlineCheckCnt;
}RC_Ctrl; 
 

uint8_t ET08_callback(uint8_t * recBuffer, uint16_t len);
void ET08Init(RC_Ctrl* RC_Ctl);
void ET08_online_protection(RC_Ctrl* RC_Ctl);
void ET08_DataUnpack(RC_Ctrl* rc_ctrl, uint8_t * recBuffer, uint16_t len );
void ET08_task(void * param);
extern RC_Ctrl rc_Ctrl;


#endif
