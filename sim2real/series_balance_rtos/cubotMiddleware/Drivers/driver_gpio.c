/**@file  bsp_usart.c
* @brief    板级支持包，串口管理器配置文件，用户回调重定义
* @details  主要包括构建串口管理器，提供串口初始化和用户回调重定义
* @author      RyanJiao  any question please send mail to 1095981200@qq.com
							 
* @date        2021-8-23
* @version     V1.1
* @copyright    Copyright (c) 2021-2121  中国矿业大学CUBOT战队
**********************************************************************************
* @attention
* 硬件平台: STM32H750VBT \n
* SDK版本：-++++
* @par 修改日志:
* <table>
* <tr><th>Date        <th>Version  <th>Author    <th>Description
* <tr><td>2021-8-12  <td>1.0      <td>RyanJiao  <td>创建初始版本
* </table>
*
**********************************************************************************
*/
#include "drv_usart.h"


struct uart_manage_obj uart_1_manage=
{
	.uart_idle_callback = NULL
};

struct uart_manage_obj uart_2_manage=
{
	.uart_idle_callback = NULL
};


uint8_t uart_manage_init(uart_manage_obj *uart_manage,UART_HandleTypeDef *huart,uint8_t *Rx_Buffer,uint8_t Rec_Buffer_Size)
{
	uart_manage->huart=(*huart);
	uart_manage->Rec_Buffer=Rx_Buffer;
	uart_manage->Rec_BUFFER_SIZE=Rec_Buffer_Size;
	
	HAL_UART_Receive_DMA(&uart_manage->huart,uart_manage->Rec_Buffer,uart_manage->Rec_BUFFER_SIZE);
	__HAL_UART_ENABLE_IT(&uart_manage->huart, UART_IT_IDLE);  //<开启中断一定要放在函数指针赋值完成之后，否则程序指向NULL，触发hard_fault();（指向NULL算运气好，不是野指针就已经万岁了）

    return 1;
} 

uint8_t uart_callback_register(uart_manage_obj *uart_manage,uart_stdmsg_callback_t rx_fn)
{
	if(rx_fn == NULL)
		return 1;
	else 
		uart_manage->uart_idle_callback = rx_fn;
	return 0;
}

void UART_IDLE_CALLBACK(uart_manage_obj *uart_manage)  
{
	if((__HAL_UART_GET_FLAG(&huart1,UART_FLAG_IDLE) != RESET))
	{  
		HAL_UART_DMAStop(&uart_manage->huart);
		__HAL_UART_CLEAR_IDLEFLAG(&uart_manage->huart);
//		if ((uart_manage->Rec_BUFFER_SIZE - uart_manage->huart.hdmarx->Instance->NDTR) == uart_manage->Data_len)
		{
			if((*uart_manage).uart_idle_callback!=NULL)
				uart_manage->uart_idle_callback(uart_manage->Rec_Buffer,uart_manage->Rec_BUFFER_SIZE); //<用户回调
		}

//	Offline_Check.Last_Update.Dbus=System.Sys_time;
		HAL_UART_Receive_DMA(&uart_manage->huart,uart_manage->Rec_Buffer,uart_manage->Rec_BUFFER_SIZE);
	}
}

/**
  * @brief 遥控器接收空闲中断回调
  * @param uart_manage:
  * @retval 
	* @usedas:
						uart1_idle_callback( task_t remote_deal,NULL,Remoter_USART1_Rec_Buffer, 30);
  */
//uint32_t uart1_idle_callback(uint8_t *buf, uint32_t Rec_Buffer_Size)
//{
//	uint8_t rx_len=0;
//	uint32_t tmp_flag = 0;
//	uint32_t temp;
//	tmp_flag =__HAL_UART_GET_FLAG(&huart1,UART_FLAG_IDLE); //获取IDLE标志位
//	
//	if((tmp_flag != RESET))//idle标志被置位
//	{ 
//		{
//			__HAL_UART_CLEAR_IDLEFLAG(&huart1);//清除标志位
//			HAL_UART_DMAStop(&huart1); //
//			temp  =  __HAL_DMA_GET_COUNTER(&hdma_usart1_rx);// 获取DMA中未传输的数据个数   
//			rx_len =  Rec_Buffer_Size - temp; //总计数减去未传输的数据个数，得到已经接收的数据个数
//		}
//		
////		remote_deal(buf,0);  //		struct Return_Struct* Remote_deal(uint8_t * Remoter_USART1_Rec_Buffer );
////		(*(uart_1_manage.task1))(buf,0);
//		if(uart_1_manage.uart_idle_callback!=NULL)
//		{
//				uart_1_manage.uart_idle_callback(buf,0);//用这个和用remote_deal(buf,0);哪个更方便还有待思考----》其实我更倾向于remote_deal，不仅少打字，还更直观。当初只是为了实验探究一下才用的这一行。
//		}
//		
//		HAL_UART_Transmit_DMA(&huart2, (uint8_t *)buf,Rec_Buffer_Size);			
//	  rx_len = 0;//清除计数
//	  HAL_UART_Receive_DMA(&huart1,buf , Rec_Buffer_Size);
//	 }
//	
//    //.......
//    return 1;
//}

void	UART1_IDLE_CALLBACK(uart_manage_obj *uart_manage)
{
	uint8_t rx_len=0;
	uint32_t tmp_flag = 0;
	uint32_t temp;
	tmp_flag =__HAL_UART_GET_FLAG(&huart1,UART_FLAG_IDLE); //获取IDLE标志位
	
	if((tmp_flag != RESET))//idle标志被置位
	{ 
		{
			__HAL_UART_CLEAR_IDLEFLAG(&huart1);//清除标志位
			HAL_UART_DMAStop(&huart1); //
			temp  =  __HAL_DMA_GET_COUNTER(&hdma_usart1_rx);// 获取DMA中未传输的数据个数   
			rx_len =  uart_manage->Rec_BUFFER_SIZE - temp; //总计数减去未传输的数据个数，得到已经接收的数据个数
		}
		
//		remote_deal(buf,0);  //		struct Return_Struct* Remote_deal(uint8_t * Remoter_USART1_Rec_Buffer );
//		(*(uart_1_manage.task1))(buf,0);
		if(uart_1_manage.uart_idle_callback!=NULL)
		{
				uart_1_manage.uart_idle_callback(uart_manage->Rec_Buffer,0);//用这个和用remote_deal(buf,0);哪个更方便还有待思考----》其实我更倾向于remote_deal，不仅少打字，还更直观。当初只是为了实验探究一下才用的这一行。
		}
		
		HAL_UART_Transmit_DMA(&huart2, uart_manage->Rec_Buffer,uart_manage->Rec_BUFFER_SIZE);			
	  rx_len = 0;//清除计数
	  HAL_UART_Receive_DMA(&huart1,uart_manage->Rec_Buffer , uart_manage->Rec_BUFFER_SIZE);
	 }
	
    //.......
 
}




















/**
  * @brief USART2空闲中断
  * @param uart_manage:
  * @retval 
	* @usedas:
						uart1_idle_callback( task_t remote_deal,NULL,Remoter_USART1_Rec_Buffer, 30);

  */

volatile uint8_t rx_len = 0;  //接收一帧数据的长度
volatile uint8_t recv_end_flag = 0; //一帧数据接收完成标志
uint8_t rx_buffer[100]={0};  //接收数据缓存数组

uint32_t uart2_idle_callback(uint8_t *buf, uint32_t Rec_Buffer_Size)
{

	uint32_t tmp_flag = 0;
	uint32_t temp;
	tmp_flag =__HAL_UART_GET_FLAG(&huart2,UART_FLAG_IDLE); //获取IDLE标志位
	
	if((tmp_flag != RESET))//idle标志被置位
	{ 
		{
			__HAL_UART_CLEAR_IDLEFLAG(&huart2);//清除标志位
			HAL_UART_DMAStop(&huart2); //
			temp  =  __HAL_DMA_GET_COUNTER(&hdma_usart2_rx);// 获取DMA中未传输的数据个数   
			rx_len =  Rec_Buffer_Size - temp; //总计数减去未传输的数据个数，得到已经接收的数据个数
		}
		
		HAL_UART_Transmit_DMA(&huart2, (uint8_t *)buf,rx_len);			
	  rx_len = 0;//清除计数
	  HAL_UART_Receive_DMA(&huart2,buf , 100);

	 }
	
    //.......
    return 1;
}


 

/**
  * @brief turn the 
  * @param uart_manage:
  * @retval 
	* @usedas:

  */
uint8_t* itoa(int  num,uint8_t* str,int radix)
{
    uint8_t index[]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";//索引表
    unsigned unum;//存放要转换的整数的绝对值,转换的整数可能是负数
    int i=0,j,k;//i用来指示设置字符串相应位，转换之后i其实就是字符串的长度；转换后顺序是逆序的，有正负的情况，k用来指示调整顺序的开始位置;j用来指示调整顺序时的交换。
 
    //获取要转换的整数的绝对值
    if(radix==10&&num<0)//要转换成十进制数并且是负数
    {
        unum=(unsigned)-num;//将num的绝对值赋给unum
        str[i++]='-';//在字符串最前面设置为'-'号，并且索引加1
    }
    else unum=(unsigned)num;//若是num为正，直接赋值给unum
 
    //转换部分，注意转换后是逆序的
    do
    {
        str[i++]=index[unum%(unsigned)radix];//取unum的最后一位，并设置为str对应位，指示索引加1
        unum/=radix;//unum去掉最后一位
 
    }while(unum);//直至unum为0退出循环
 
    str[i]='\0';//在字符串最后添加'\0'字符，c语言字符串以'\0'结束。
 
    //将顺序调整过来
    if(str[0]=='-') k=1;//如果是负数，符号不用调整，从符号后面开始调整
    else k=0;//不是负数，全部都要调整
 
    uint8_t temp;//临时变量，交换两个值时用到
    for(j=k;j<=(i-1)/2;j++)//头尾一一对称交换，i其实就是字符串的长度，索引最大值比长度少1
    {
        temp=str[j];//头部赋值给临时变量
        str[j]=str[i-1+k-j];//尾部赋值给头部
        str[i-1+k-j]=temp;//将临时变量的值(其实就是之前的头部值)赋给尾部
    }
 
    return str;//返回转换后的字符串
 
}


