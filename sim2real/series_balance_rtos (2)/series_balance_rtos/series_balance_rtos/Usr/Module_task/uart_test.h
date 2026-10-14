#ifndef RTOS_TASK_H_
#define RTOS_TASK_H_

#include "usart.h"
#include "uart_base.h"

extern UART_RxBuffer uart5_buffer;
extern UART_RxBuffer uart4_buffer;

uint8_t uart4_callback(uint8_t * recBuffer, uint16_t len);
uint8_t uart5_callback(uint8_t * recBuffer, uint16_t len);

#endif