#include "stm32h7xx_hal.h"
#include "usart.h"

#ifndef __HI04_H
#define __HI04_H

#include "uart_base.h"

#define HI04_rxBufferLengh  58

typedef struct {       
	 uint8_t   tag;
	 uint16_t  status;       
	 int8_t    temperate;       
	 uint16_t  pps;      
	 int16_t   air_press;      
	 int16_t   heave; 
	 int16_t   gyr_b[3];	
	 int16_t   acc_b[3]; 
	 int16_t   mag_b[3]; 
	 int32_t   roll;
	 int32_t   pitch;
	 int32_t   yaw;
	 int16_t   quat[4];
	struct {       
			float  roll_speed;
			float  pitch_speed;       
			float  yaw_speed;       
			float  roll;      
			float  pitch;
		    float  pitch_last;
			float  yaw; 
		    float  yaw_last; 
			float  yaw_total;
	}gyro_data; 
	float accel[3]; 
	}HI04_data; 
 
uint8_t HI04_callback(uint8_t *recBuffer, uint16_t len);
uint8_t HI04_data_receive(uint8_t * recBuffer);
void HI04_modify(int8_t init_check);
extern UART_RxBuffer uart8_buffer;
extern HI04_data HI04_data_t;


#endif
	