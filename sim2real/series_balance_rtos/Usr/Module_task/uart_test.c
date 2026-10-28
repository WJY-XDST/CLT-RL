#include "freertos.h"
#include "gpio.h"
#include "usart.h"
#include "main.h"
#include "queue.h"
#include "cmsis_os.h"
#include "uart_base.h"

uint8_t txbuffer[2]={1,2};
uint8_t txbuffer4[2]={4,4};
int check4=0,check5=0;

uint8_t Usart5_RxBuffer[1]__attribute__((at(0x24001000)));
uint8_t Usart5_TxBuffer[1]__attribute__((at(0x24001800)));

uint8_t Usart4_RxBuffer[1]__attribute__((at(0x24003000)));
uint8_t Usart4_TxBuffer[1]__attribute__((at(0x24003800)));

UART_RxBuffer uart5_buffer={
	.Data = Usart5_RxBuffer,
	.Size = 1
};

UART_RxBuffer uart4_buffer={
	.Data = Usart4_RxBuffer,
	.Size = 1
};

void led_task(void const * argument)
{
	
 while(1)
 {
	 
    HAL_UART_Transmit_DMA(&huart5,txbuffer,2);
	HAL_Delay(50);
	HAL_UART_Transmit_DMA(&huart4,txbuffer4,2);
	HAL_Delay(50); 
}
 }

 
uint8_t uart4_callback(uint8_t * recBuffer, uint16_t len)
{
 check4=recBuffer[0]+recBuffer[1];
 
}

uint8_t uart5_callback(uint8_t * recBuffer, uint16_t len)
{
 check5=recBuffer[0]+recBuffer[1];

}
	
	
	
	