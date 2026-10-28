#ifndef __DJI_MOTOR_H__
#define __DJI_MOTOR_H__

#include "stm32h7xx_hal.h"
#include "can_base.h"
#include "linux_list.h" 


#define K_ECD_TO_ANGLE 0.043945f  		//< 角度转换编码器刻度的系数：360/8192
#define ECD_RANGE_FOR_3508 8191				//< 编码器刻度值为0-8191
#define CURRENT_LIMIT_FOR_3508 16000   //< 控制电流范围为正负16384
#define ECD_RANGE_FOR_6020 8191				//< 编码器刻度值为0-8191
#define CURRENT_LIMIT_FOR_6020 29000   //< 控制电流范围为正负30000
#define ECD_RANGE_FOR_2006 8191				//< 编码器刻度值为0-8191
#define CURRENT_LIMIT_FOR_2006 10000   //< 控制电流范围为正负16384
#define CURRENT_LIMIT_FOR_DM 16000   //< 控制电流范围为正负16384
#define ECD_RANGE_FOR_DM 8191				//< 编码器刻度值为0-8191



/**
  * @brief  定义电机种类，用于发送函数的选择
	* @note   GM6020的反馈报文ID为 0x205-0x20B 之间
	* @note   M3508的反馈报文ID为 0x201-0x208 之间
  */
typedef enum 
{
	Motor3508 = 0x00U,
	Motor6020 = 0x01U,
	Motor2006 = 0x02U,
  DMmotor   = 0x03U
}MotorType;


/**
  * @brief  电机动态数据，电机运行中产生的数据，由CAN中断回调更新
  */
typedef struct
{
	int16_t  Ecd;         	//< 当前编码器返回值
	int16_t  SpeedRPM;			//< 每分钟所转圈数
	int16_t  TorqueCurrent; //< 反馈力矩
	uint8_t  Temperature;	  //< 温度
	
	int16_t  RawEcd;				//< 原始编码器数据
	int16_t  LastEcd;			  //< 上一时刻编码器返回值			
	float    Angle;					//< 解算后的编码器角度
	int16_t  AngleSpeed;	  //< 解算后的编码器角速度	
	int32_t  RoundCnt;			//< 累计转动圈数
	int32_t  TotalEcd;			//< 编码器累计增量值
	int32_t  TotalAngle;		//< 累计旋转角度

	int16_t  Target;				//< 电机的期望参数
	int32_t  Output;  			//< 电机输出值，通常为电流和电压	
	float CanEcd[20] ;
	float CanAngleSpeed[20] ;
	float LvboAngle;
	int16_t  LvboEcd;
	int16_t  LvboSpeedRPM;
	
}MotorData;


/**
  * @brief   电机参数，在初始化函数中确定
  */
typedef struct 														
{
	uint8_t  CanNumber;			 										//< 电机所使用的CAN端口号
	uint16_t CanId;			 												//< 电机ID	
	uint8_t  MotorType;			 										//< 电机类型	
	uint16_t EcdOffset;	 									  		//< 电机初始零点
	uint16_t EcdFullRange;											//< 编码器量程
	int16_t  CurrentLimit;			 								//< 电调能承受的最大电流
}MotorParam;



/**
  * @brief     将电机的待发送数据填入CAN发送缓存区
  * @param[in] motor_data     电机动态数据结构体，只需要发送，不修改数据，不需要传入指针、
  * @param[in] id             电机标识符ID
  */
typedef uint8_t (*CAN_FillMotorData)(CAN_Object can, MotorData motor_data, uint16_t id); //< 电机发送回调


/**
  * @brief  筛选CAN数据，并更新电机动态数据的回调函数
  */
typedef uint8_t (*Motor_DataUpdate)(MotorData* motor_data, CAN_RxBuffer rxBuffer); //< 电机发送回调



/**
  * @brief  电机的数据和参数，以及两个不同电机之间略有区别的成员函数
  */
typedef struct 
{
	list_t             list;			 											//< 链表指针，创建指向自己的初始链表，并通过注册函数拓展成循环链表
	MotorData          Data;   	 											  //< 电机动态数据，工作中更新
	MotorParam         Param;			 											//< 电机参数，在初始化时设置
	
	Motor_DataUpdate   MotorUpdate;											//< 更新电机运行数据的函数指针
	CAN_FillMotorData  FillMotorData;										//< 对不同发送ID的CAN发送缓存区填入待发送数据的函数指针
}Motor;



/**
  * @brief  		电机初始化
	* @note 			GM6020电机ID为001时，反馈报文为0x205,电机ID为111时（7），反馈报文ID为0x20B
  * @param[in]  motor      电机数据结构体
  * @param[in]  ecd_Offset  编码器初始值
  * @param[in]  type       电机类型 (枚举)
  * @param[in]  canx       CAN端口号 (枚举)
  * @param[in]  id         电机ID标识符
  */
void MotorInit(Motor* motor, uint16_t ecd_Offset, MotorType type, CanNumber canx, uint16_t id);


/**
  * @brief  		电机的接收回调业务逻辑，在CAN接收中断中判断完电机ID后调用
  * @param[in]  motor      电机数据结构体
  * @param[in]  rxBuffer   CAN接收数据缓存区
  */
void MotorRxCallback(CAN_Object canx, CAN_RxBuffer rxBuffer);



/**
  * @brief  		获得电机结构体中的ID
  * @param[in]  motor   电机数据结构体
  */
uint16_t MotorReturnID(Motor motor);



/**
  * @brief  		将motor_data.Output填入发送缓存区等待发送的函数。
	* @note			  电机的反馈报文ID在不同区间内，控制电机报文的id不同，故需要编写此函数。
  * @param[in]  motor      电机数据结构体
  * @param[in]  output     待输出的控制量
*/
void MotorFillData(Motor* motor, int32_t output);




/**
  * @brief  		将特定ID的CAN_TxBuffer发送出去
	* @note 			电机的反馈ID设置，影响电机的控制ID：	
	* @note 			大疆电机3508动力电机作为底盘时通常使用0x200作为控制ID，尽量将四个底盘电机ID设置为0x201-0x204或者0x205-0x208，使其在一个控制ID下
  * @param[in]  can						  		CAN设备结构体
  * @param[in]  IDforTxBuffer  		  电机控制ID
  */
uint16_t MotorCanOutput(CAN_Object can, int16_t IDforTxBuffer);


extern int Q_index;
extern float Angle_sum;
extern float Speed_sum;

uint8_t CAN1_rxCallBack(CAN_RxBuffer* rxBuffer);
uint8_t CAN2_rxCallBack(CAN_RxBuffer* rxBuffer);
void motor_data_modify_task(void *param);


#endif // __MOTOR_H__
