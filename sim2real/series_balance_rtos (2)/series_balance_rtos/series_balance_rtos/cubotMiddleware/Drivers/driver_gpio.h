#ifndef DRV_USART_H
#define DRV_USART_H
#include "stm32h7xx_hal.h"
#include "usart.h"



/**
  * @brief   串口用户回调函数解算
  */
typedef uint8_t (*uart_stdmsg_callback_t)(uint8_t *buff , uint16_t len);//相当于声明了一个函数指针类型

/**
  * @brief   串口管理器结构体
  */
typedef struct uart_manage_obj
{
    UART_HandleTypeDef huart;  
    uint8_t *Rec_Buffer;
    uint16_t  Rec_BUFFER_SIZE;
	  uint16_t Data_len;
    uart_stdmsg_callback_t uart_idle_callback;
}uart_manage_obj; 

/**
  * @brief  串口设备初始化，通过串口管理器执行初始化赋值 
  * @param uart_manage:
  * @retval 
  */
uint8_t uart_manage_init(uart_manage_obj *uart_manage,UART_HandleTypeDef *huart,uint8_t *Rx_Buffer,uint8_t Rec_Buffer_Size);

/**
  * @brief  串口用户回调函数初始化 ，更改串口管理器中回调函数指针的赋值
  * @param uart_manage:
  * @retval 
  */
uint8_t uart_callback_register(uart_manage_obj *uart_manage,uart_stdmsg_callback_t rx_fn);

/**
  * @brief  串口设备中断函数，执行中断DMA操作，调用串口用户回调函数 
  * @param uart_manage:
  * @retval 
  */
void UART_IDLE_CALLBACK(uart_manage_obj *uart_manage);

/**
  * @brief  整型转换成字符串ASCii码
  * @param  num 整型数 radix整型数进制
  * @retval str 字符串
  */
uint8_t* itoa(int num,uint8_t* str,int radix);




void	UART1_IDLE_CALLBACK(uart_manage_obj *uart_manage);




extern struct uart_manage_obj uart_1_manage;
extern struct uart_manage_obj uart_2_manage;


#endif 

