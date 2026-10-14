#include "A1_stm32.h"
#include "filter.h"
#include <math.h>
#include <string.h>
#include "vmc.h"
#include "et08.h"
#include "all_init.h"

/*电机下电判断车被打死*/


//< 收发数组
uint8_t Usart4_RxBuffer[USART_RXBUF_SIZE]__attribute__((at(0x24002100)));
uint8_t Usart4_TxBuffer[USART_TXBUF_SIZE]__attribute__((at(0x24002800)));
uint8_t Usart5_RxBuffer[USART_RXBUF_SIZE]__attribute__((at(0x24003100)));
uint8_t Usart5_TxBuffer[USART_TXBUF_SIZE]__attribute__((at(0x24003800)));

UART_RxBuffer uart4_buffer={
	.Data = Usart4_RxBuffer,
	.Size = USART_RXBUF_SIZE
};
UART_RxBuffer uart5_buffer={
	.Data = Usart5_RxBuffer,
	.Size = USART_RXBUF_SIZE
};

//电机收发结构体变量声明
MOTOR_send Motor_zero_send_left = 
{
	.motor_send_data=
	{
		0
	},
	.hex_len=34, 
	.id     =MOTOR_ZERO_ID,  //0xbb为广播
	.mode   =10, //闭环伺服控制
	.T      =0,   //># 单位：Nm, |T|<128
	.W      =0, //># 单位：rad/s, |W|<256
	.Pos    =0,   //># 单位：rad, |Pos|<823549 减速比：9.1
	.K_P    =0.2,   //># 0<K_P<16
	.K_W    =3    //># 0<K_W<32
};


//电机收发结构体变量声明
MOTOR_send Motor_one_send_left = 
{
	.motor_send_data=
	{
		0
	},
	.hex_len=34, 
	.id     =MOTOR_ONE_ID,  //0xbb为广播
	.mode   =10, //闭环伺服控制
	.T      =0,   //># 单位：Nm, |T|<128
	.W      =0, //># 单位：rad/s, |W|<256
	.Pos    =0,   //># 单位：rad, |Pos|<823549 减速比：9.1
	.K_P    =0.2,   //># 0<K_P<16
	.K_W    =3    //># 0<K_W<32
};
//电机收发结构体变量声明
MOTOR_send Motor_zero_send_right = 
{
	.motor_send_data=
	{
		0
	},
	.hex_len=34, 
	.id     =MOTOR_ZERO_ID,  //0xbb为广播
	.mode   =10, //闭环伺服控制
	.T      =0,   //># 单位：Nm, |T|<128
	.W      =0, //># 单位：rad/s, |W|<256
	.Pos    =0,   //># 单位：rad, |Pos|<823549 减速比：9.1
	.K_P    =0.2,   //># 0<K_P<16
	.K_W    =3    //># 0<K_W<32
};

//电机收发结构体变量声明
MOTOR_send Motor_one_send_right = 
{
	.motor_send_data=
	{
		0
	},
	.hex_len=34, 
	.id     =MOTOR_ONE_ID,  //0xbb为广播
	.mode   =10, //闭环伺服控制
	.T      =0,   //># 单位：Nm, |T|<128
	.W      =0, //># 单位：rad/s, |W|<256
	.Pos    =0,   //># 单位：rad, |Pos|<823549 减速比：9.1
	.K_P    =0.2,   //># 0<K_P<16
	.K_W    =3    //># 0<K_W<32
};







MOTOR_recv Motor_zero_receive_left;
MOTOR_recv Motor_one_receive_left;
MOTOR_recv Motor_zero_receive_right;
MOTOR_recv Motor_one_receive_right;

/**
  * @brief A1初始化函数（pitch1，pitch2）
  */
void A1_init()
{
	Motor_VelocityMode(&Motor_zero_send_left,3.5);		//进入速度模式
	Motor_VelocityMode(&Motor_one_send_left,-3.5);
	Motor_VelocityMode(&Motor_zero_send_right,3.5);	
	Motor_VelocityMode(&Motor_one_send_right,-3.5);
	//Motor_ZeroTorque(&Motor_zero_send_left);
    //Motor_ZeroTorque(&Motor_one_send_left);
	//Motor_ZeroTorque(&Motor_zero_send_right);
	//Motor_ZeroTorque(&Motor_one_send_right);
	

}
//输出阻尼
void A1_Dampingtorque()
{

	if(rc_Ctrl.rc.s1==2&&rc_Ctrl.isOnline ==1)
	{
		A1_init();
	}
	else
    {
		A1_DampingMode(&Motor_zero_send_left);
		A1_DampingMode(&Motor_one_send_left);
		A1_DampingMode(&Motor_zero_send_right);
		A1_DampingMode(&Motor_one_send_right);
	}
	
}
/**
  * @brief A1发送函数
  */
uint8_t A1_send(UART_HandleTypeDef* huart_x, MOTOR_send* pMotor_s)
{

	if(huart_x == &huart4)
	{
		modify_data(pMotor_s,Usart4_TxBuffer);
		HAL_UART_Transmit_DMA(huart_x,Usart4_TxBuffer,USART_TXBUF_SIZE);
	}
	if(huart_x == &huart5)
	{
		modify_data(pMotor_s,Usart5_TxBuffer);
		HAL_UART_Transmit_DMA(huart_x,Usart5_TxBuffer,USART_TXBUF_SIZE);
	}
	return 1;
}

/**
  * @brief 电机进入零力矩模式
  */
void Motor_ZeroTorque(MOTOR_send* pMotor_s)
{
	pMotor_s->T = 0;
	pMotor_s->W = 0;
	pMotor_s->Pos = 0;
	pMotor_s->K_P = 0;
	pMotor_s->K_W = 0;
}


/**
  * @brief A1阻尼模式
  */
void A1_DampingMode(MOTOR_send* pMotor_s)
{
	pMotor_s->T = 0;
	pMotor_s->W = 0;
	pMotor_s->Pos = 0;
	pMotor_s->K_P = 0;
	pMotor_s->K_W = 1;
}


/**
  * @brief 电n机进入速度控制模式
  */
void Motor_VelocityMode(MOTOR_send* pMotor_s, float velocity)
{
	pMotor_s->T = 0;
	pMotor_s->W = velocity * 9.1;	 //< 单位：rad/s,
	pMotor_s->Pos = 0;
	pMotor_s->K_P = 0;
	pMotor_s->K_W = 3.5;
}

/**
  * @brief 电机进入位置控制模式，控制电机输出轴的角度
  */
void Motor_DegPositionMode(MOTOR_send* pMotor_s, float angle) 
{
	pMotor_s->T = 0;
	pMotor_s->W = 0;	 										//< 单位：rad/s,
	pMotor_s->Pos = angle * AtR * 9.1f;   //< 单位是rad，需要先将角度转换为弧度，再将弧度乘减速比得到出轴的角度
	pMotor_s->K_P = 0.2;   //< Kp < 16
	pMotor_s->K_W = 3;		 //< Kd < 32
}






/**
  * @brief 电机进入力矩控制模式，空载安全力矩为0.05 单位Nm T<128
  */
void Motor_TorqueMode(MOTOR_send* pMotor_s, float tau) 
{
	pMotor_s->T = tau;
	pMotor_s->W = 0;	 	
	pMotor_s->Pos = 0;  
	pMotor_s->K_P = 0;
	pMotor_s->K_W = 0;
}


//ID读取回调函数，当前使用串口4,5
MOTOR_recv* Motor_receive_callback(uint8_t ID, UART_HandleTypeDef* huart)
{
	
	if(huart == &huart5)
	{
		switch(ID)
		{
			case 0:
				return &(Motor_zero_receive_left);
			case 1:
				return &(Motor_one_receive_left);
			default:
				return NULL;
		}
		
	}
		if(huart == &huart4)
	{
		switch(ID)
		{
			case 0:
				return &(Motor_zero_receive_right);
			case 1:
				return &(Motor_one_receive_right);
			default:
				return NULL;
		}
		
	}
}

//串口空闲中断中调用
void Motor_receive_test(UART_HandleTypeDef* huart)
{

	if(huart == &huart4)
		A1_data_receive(huart, Motor_receive_callback, Usart4_RxBuffer, USART_RXBUF_SIZE);
	if(huart == &huart5)
		A1_data_receive(huart, Motor_receive_callback, Usart5_RxBuffer, USART_RXBUF_SIZE);

}



//stm32 HAL库 A1电机通信驱动
//调用接收函数时，需要注册回调函数 回调函数参数为接收到的电机ID 返回对应电机的结构体

uint8_t A1_data_send(UART_HandleTypeDef* huart_x,MOTOR_send* pMotor_s,uint8_t* pBuffer,uint8_t len)
{
	//调用A1协议封包函数 将数据装入发送协议包结构体
	modify_data(pMotor_s,pBuffer);
	// 发送
	HAL_UART_Transmit_DMA(huart_x,pBuffer,len);
	// 执行成功
	return 1;
}


uint8_t A1_data_receive(UART_HandleTypeDef* huart_x,MOTOR_recv* callback_fun(uint8_t, UART_HandleTypeDef*),uint8_t* pBuffer,uint8_t len)
{ 
	uint8_t receive_id;
	MOTOR_recv* pMotor_r;
	//解包 将pBuffer中的数据解入pMotor_r
	receive_id = check_data_forID(pBuffer);
	//执行回调函数，获取对应ID的结构体指针
	pMotor_r = callback_fun(receive_id, huart_x);
	//将数据包数据解到结构体中
	extract_data(pMotor_r,pBuffer);
}


uint8_t A1_fake_callback_left(uint8_t *recBuffer, uint16_t len)
{
	Motor_receive_test(&huart5);
	return 0;
}

uint8_t A1_fake_callback_right(uint8_t *recBuffer, uint16_t len)
{
	Motor_receive_test(&huart4);
	return 0;
}


