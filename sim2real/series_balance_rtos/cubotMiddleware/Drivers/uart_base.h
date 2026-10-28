#ifndef UART_BASE_H
#define UART_BASE_H
#include "stm32h7xx_hal.h"
#include "usart.h"

/*和freertos相关*/
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

/**
  * @brief   串口用户回调函数解算
  * @param[in]  rxBuffer		串口接受数据的数组首地址
  * @param[in]  size	    	串口接收数据的数组长度
  */
typedef uint8_t (*UART_RxIdleCallback)(uint8_t *rxBuffer, uint16_t size); ///<相当于声明了一个函数指针类型


/**
	* @brief	UART发送缓冲区
	*/
typedef struct
{
    uint8_t* Data;
	uint16_t Size;
}UART_RxBuffer;


/**
	* @brief	UART接收缓冲区
	*/
typedef struct
{
	uint8_t* Data;
	uint16_t DataSize;
}UART_TxBuffer;


/**
  * @brief   串口结构体,包含句柄和接收回调
  */
typedef struct 
{
	UART_HandleTypeDef* Handle;
	UART_RxIdleCallback RxIdleCallback;
    QueueHandle_t g_xQueueUart;	
}UART_Object;


/**
  * @brief   串口初始化，将句柄和接收回调拷贝至串口结构体
  * @param[in]  handle		        串口句柄
  * @param[in]  rxIdleCallback		接收回调函数
  */
void UARTx_Init(UART_HandleTypeDef* handle, UART_RxIdleCallback rxIdleCallback);


/**
  * @brief  串口管理器结构体参数已经预先填写好的串口设备初始化
  * @param[in]  uart		    串口结构体, 标明串口句柄
  * @param[in]  rxBuffer		接收缓存区,用户定义
  * @retval 
  */
void UART_Receive_DMA(UART_Object* uart, UART_RxBuffer* rxBuffer);
void UART_ENABLE_IT(UART_Object* uart, UART_RxBuffer* rxBuffer);

/**
  * @brief  串口调用dma发送数据，数据需拆分为字节
  * @param[in]  uart		    串口结构体，标明使用的串口号
  * @param[in]  txBuffer		发送缓存区.用户定义
  * @retval 
  */
uint32_t UART_Send(UART_Object* uart, UART_TxBuffer* txBuffer);


/**
  * @brief  串口设备中断函数，执行中断DMA操作，调用串口用户回调函数，需要将其添加到stm32h7xx_it.c中
  * @param[in]  uart		    串口结构体, 标明串口句柄和回调函数
  * @param[in]  rxBuffer		发送缓存区, 用户定义。注意！ 此处的rxBuffer应该与 UART_Open中调用rxBuffer一致
  * @retval 
  */
void UART_Idle_Handler(UART_Object* uart, UART_RxBuffer* rxBuffer);


/**
  * @brief  整型转换成字符串ASCii码
  * @param  num 整型数 radix整型数进制
  * @retval str 字符串
  */
uint8_t* itoa(int num,uint8_t* str, int radix);
void ANO_V6_Send_Up_Computer(int16_t user1,int16_t user2,int16_t user3,int16_t user4,int16_t user5,int16_t user6);
extern uint8_t data_to_send_V6[20]__attribute__((at(0x24002000)));

extern UART_Object uart1;
extern UART_Object uart2;
extern UART_Object uart3;
extern UART_Object uart4;
extern UART_Object uart5;
extern UART_Object uart6;
extern UART_Object uart7;
extern UART_Object uart8;

#endif 


