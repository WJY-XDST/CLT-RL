#ifndef __CAN_BASE_H
#define __CAN_BASE_H
#include "fdcan.h"
#include "linux_list.h"

/*和freertos相关*/
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
typedef enum{
    
	CAN1= 0x11U,             //无符号
	CAN2= 0x12U,
}CanNumber;



/**
	* @brief	CAN接收缓冲区
	*/
typedef struct
{
	FDCAN_RxHeaderTypeDef	            Header;
	uint8_t								Data[8];
}CAN_RxBuffer;

/**
	* @brief	CAN发送缓冲区
	*/
typedef struct
{
	FDCAN_TxHeaderTypeDef txHeader;
	uint32_t							Identifier;
	uint8_t								Data[8];
}CAN_TxBuffer;


/**
  * @brief   CAN用户回调函数
	* @param[in] rxBuffer 	CAN接收缓冲区
  */
typedef uint8_t(*CAN_RxCpltCallback)(CAN_RxBuffer* rxBuffer);

/**
	* @brief	CAN设备对象
	*/
typedef struct
{
	FDCAN_HandleTypeDef* 	Handle;
	list_t					DevicesList;
	CAN_RxCpltCallback		RxCpltCallback;
	QueueHandle_t g_xQueueCan;
}CAN_Object;


/**
  * @brief  CAN初始化，将句柄和接收回调拷贝至CAN设备结构体
  */
void CANx_Init(FDCAN_HandleTypeDef* handle, CAN_RxCpltCallback rxCallback);

/**
  * @brief 打开CAN设备，配置过滤器为空，使能fifo0接收到新信息中断，注册用户回调
  * @param[in]	can	CAN设备
  */
void CAN_Open(CAN_Object* can);


/**
  * @brief 通过CAN设备发送数据
  *	@param[in] txBuffer CAN的发送缓冲区
	* 
  */
uint8_t CAN_Send(CAN_Object* can,CAN_TxBuffer *txbuffer );



extern CAN_Object can1;//
extern CAN_Object can2;


#endif
