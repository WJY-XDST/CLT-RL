//	/**@file  dr16.c
//	* @brief    设备层
//	* @details  主要包括构建串口管理器，提供串口初始化和用户回调重定义
//	* @author      RyanJiao  any question please send mail to 1095981200@qq.com
//	* @date        2021-10-9
//	* @version     V1.0
//	* @copyright    Copyright (c) 2021-2121  中国矿业大学CUBOT战队
//	**********************************************************************************
//	* @attention
//	* 硬件平台: STM32H750VBT \n
//	* SDK版本：-++++
//	* @par 修改日志:
//	* <table>
//	* <tr><th>Date        <th>Version  <th>Author    <th>Description
//	* <tr><td>2021-8-12  <td>1.0      <td>RyanJiao  <td>创建初始版本
//	* </table>
//	*
//	**********************************************************************************
//	 ==============================================================================
//														How to use this driver  
//	 ==============================================================================
//	 

//		********************************************************************************
//		* @attention
//		* 硬件平台: STM32H750VBT \n
//		* SDK版本：-++++
//		* if you had modified this file, please make sure your code does not have many 
//		* bugs, update the version NO., write dowm your name and the date, the most
//		* important is make sure the users will have clear and definite understanding 
//		* through your new brief.
//		********************************************************************************
//	*/

//	#include "dr16.h"
//  /*和freertos相关*/
//  #include "freertos.h"
//  #include "FreeRTOS.h"
//  #include "task.h"
//  #include "queue.h"
//	#define myABS(x) ( (x)>0?(x):-(x) )
//	RC_Ctrl rc_Ctrl={
//		.isUnpackaging = 0,
//		.isOnline = 0
//		
//	};

//	/**
//		* @brief  创建dr16的接收机缓存数组
//		*/
//	uint8_t DR16_recData[DR16_rxBufferLengh]__attribute__((at(0x24006000)));


//	/**
//		* @brief  创建dr16串口缓存区数据结构
//		*/
//	UART_RxBuffer uart1_buffer={
//		.Data = DR16_recData,
//		.Size = DR16_rxBufferLengh
//	};

//	/**
//		* @brief  初始化接收机数据类型的数据，将杆量和按键信息归零
//		*/
//	void DR16Init(RC_Ctrl* RC_Ctl)
//	{
//		RC_Ctl->rc.ch0=1024;
//		RC_Ctl->rc.ch1=1024;
//		RC_Ctl->rc.ch2=1024;
//		RC_Ctl->rc.ch3=1024;
//		RC_Ctl->rc.s1=3;
//		RC_Ctl->rc.s2=3;
//		RC_Ctl->rc.sw=1024;
//		RC_Ctl->mouse.x=0;
//		RC_Ctl->mouse.y=0;
//		RC_Ctl->mouse.z=0;		
//		RC_Ctl->OneShoot = 0;
//		RC_Ctl->key_Q_flag=0;
//		RC_Ctl->key_E_flag=0;    //< 上电关弹舱
//		RC_Ctl->key_R_flag=0;
//		RC_Ctl->key_F_flag=0;
//		RC_Ctl->key_G_flag=0;
//		RC_Ctl->key_Z_flag=0;
//		RC_Ctl->key_X_flag=0;
//		RC_Ctl->key_C_flag=0;
//		RC_Ctl->key_V_flag=0;
//		RC_Ctl->key_B_flag=0;
//		RC_Ctl->Chassis_Y_Integ=0;//斜坡积分变量
//		RC_Ctl->Chassis_X_Integ=0;
//		RC_Ctl->ShootNumber=9;
//		RC_Ctl->Cruise_Mode = 0;
//	}


//	/**
//		* @brief  创建dr16的接收机缓存数组
//		*/
//	uint8_t DR16_callback(uint8_t * recBuffer, uint16_t len)
//	{
//		xQueueSendFromISR(uart1.g_xQueueUart,&(recBuffer), 0);
//		return 0;
//	}
//	
//	/**
//	* @brief  创建dr16的处理任务
//	*/
//	void DR16_task(void * param)
//	{
//		uint8_t * RecBuffer;
//		while(1)
//		{
//			xQueueReceive(uart1.g_xQueueUart,&RecBuffer,portMAX_DELAY);
//			DR16_DataUnpack(&rc_Ctrl, RecBuffer,DR16_rxBufferLengh);  //< callback函数由格式限制
//		}
//	
//	}

//	/**
//		* @brief  创建dr16的接收机缓存数组, 并对全局变量rc_Ctrl赋值，以供其他函数调用
//		*/
//	
//	void DR16_DataUnpack(RC_Ctrl* rc_ctrl, uint8_t * recBuffer, uint16_t len )
//	{ 
//		rc_ctrl->isUnpackaging = 1;					//< 解算期间不允许读取数据
//		uint8_t correct_num=0;	
//		correct_num=0;
//		if(((recBuffer[0] | (recBuffer[1] << 8)) & 0x07ff)<=1684 && ((recBuffer[0] | (recBuffer[1] << 8)) & 0x07ff)>=364)
//			correct_num++;
//		if((((recBuffer[1] >> 3) | (recBuffer[2] << 5)) & 0x07ff)<=1684 && (((recBuffer[1] >> 3) | (recBuffer[2] << 5)) & 0x07ff)>=364)
//			correct_num++;
//		if((((recBuffer[2] >> 6) | (recBuffer[3] << 2) |(recBuffer[4] << 10)) & 0x07ff)<=1684 && (((recBuffer[2] >> 6) | (recBuffer[3] << 2) |(recBuffer[4] << 10)) & 0x07ff)>=364)
//			correct_num++;
//		if((((recBuffer[4] >> 1) | (recBuffer[5] << 7)) & 0x07ff)<=1684 && (((recBuffer[4] >> 1) | (recBuffer[5] << 7)) & 0x07ff)>=364)
//			correct_num++;
//		if((((recBuffer[5] >> 4)& 0x000C) >> 2)==1 || (((recBuffer[5] >> 4)& 0x000C) >> 2)==2 || (((recBuffer[5] >> 4)& 0x000C) >> 2)==3)
//			correct_num++;
//		if(((recBuffer[5] >> 4)& 0x0003)==1 || ((recBuffer[5] >> 4)& 0x0003)==2 || ((recBuffer[5] >> 4)& 0x0003)==3)
//			correct_num++;
//		if(correct_num==6)																																												//< 数据完整性验证 
//		{
//			rc_ctrl->rc.ch0 = (recBuffer[0]| (recBuffer[1] << 8)) & 0x07ff; 																					//< Channel 0   高8位与低3位
//			rc_ctrl->rc.ch1 = ((recBuffer[1] >> 3) | (recBuffer[2] << 5)) & 0x07ff; 																	//< Channel 1   高5位与低6位
//			rc_ctrl->rc.ch2 = ((recBuffer[2] >> 6) | (recBuffer[3] << 2) |(recBuffer[4] << 10)) & 0x07ff; 						//< Channel 2
//			rc_ctrl->rc.ch3 = ((recBuffer[4] >> 1) | (recBuffer[5] << 7)) & 0x07ff; 																	//< Channel 3
//			rc_ctrl->rc.s1 = ((recBuffer[5] >> 4)& 0x000C) >> 2; 																											//!< Switch left
//			rc_ctrl->rc.s2 = ((recBuffer[5] >> 4)& 0x0003); 																													//!< Switch right
//			rc_ctrl->rc.sw=(uint16_t)(recBuffer[16]|(recBuffer[17]<<8))&0x7ff; 
//				
//			if((rc_ctrl->rc.ch0>1020)&&(rc_ctrl->rc.ch0<1028))          //遥控器零飘
//				rc_ctrl->rc.ch0=1024;
//			if((rc_ctrl->rc.ch1>1020)&&(rc_ctrl->rc.ch1<1028))
//				rc_ctrl->rc.ch1=1024;
//			if((rc_ctrl->rc.ch2>1020)&&(rc_ctrl->rc.ch2<1028))
//				rc_ctrl->rc.ch2=1024;
//			if((rc_ctrl->rc.ch3>1020)&&(rc_ctrl->rc.ch3<1028))
//				rc_ctrl->rc.ch3=1024;
//			}
//		rc_Ctrl.onlineCheckCnt=0;

//		}

///**
//* @brief  按键消抖，检测是否为有效按下
//*/
//void PC_keybroad_filter(RC_Ctrl* RC_Ctl)
//{
//	static uint16_t key_W_cnt,key_A_cnt,key_S_cnt,key_D_cnt,key_ctrl_cnt,
//								 key_shift_cnt,mouse_press_l_cnt,mouse_press_r_cnt,
//								 key_C_cnt,key_F_cnt,key_G_cnt,key_Q_cnt,key_E_cnt,
//									key_Z_cnt,key_V_cnt,key_X_cnt,key_B_cnt,key_R_cnt;
//		/*   支持连续按 W A S D   */
//		//< key_W
//	if(RC_Ctl->key_W==1) 
//	{
//		key_W_cnt++;
//		if(key_W_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_W_flag=1;	
//		}	 
//	}   
//	else
//	{
//		RC_Ctl->key_W_flag=0;	
//		key_W_cnt=0;	
//	}	
//			//key_A
//	if(RC_Ctl->key_A==1) 
//	{
//		key_A_cnt++;
//		if(key_A_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_A_flag=1;
//		}	
//	}

//	else
//	{	
//		key_A_cnt=0;	
//		RC_Ctl->key_A_flag=0;
//	}
//	//key_S
//	if(RC_Ctl->key_S==1) 
//	{
//		key_S_cnt++;
//		if(key_S_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_S_flag=1;	
//		}			
//	}	
//	else
//	{
//		key_S_cnt=0;
//		RC_Ctl->key_S_flag=0;
//	}		
//	//key_D
//	if(RC_Ctl->key_D==1) 
//	{
//		key_D_cnt++;
//		if(key_D_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_D_flag=1;			
//		}	
//	}		
//	else
//	{
//		key_D_cnt=0;
//		RC_Ctl->key_D_flag=0;
//	}
//	
//		//key_B
//	if(RC_Ctl->key_B==1) 
//	{
//		key_B_cnt++;
//		if(key_B_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_B_flag=1;
//		}	
//	}	
//	else
//	{
//		key_B_cnt=0;
//		RC_Ctl->key_B_flag=0;
//	}
//	
//	//key_C
//	if(RC_Ctl->key_C==1) 
//	{
//		key_C_cnt++;
//		if(key_C_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_C_flag = 1;
//		}	 
//	}  
//	else
//	{
//		key_C_cnt=0;
//		RC_Ctl->key_C_flag=0;
//	}	
//		//key_R
//	if(RC_Ctl->key_R==1) 
//	{
//		key_R_cnt++;
//		if(key_R_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_R_flag++;			
//		}	 
//	}   else
//	 {
//		key_R_cnt=0;	
//	 }
//	 
//	//key_F
//	if(RC_Ctl->key_F==1) 
//	{
//		key_F_cnt++;
//		if(key_F_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_F_flag=1;
//		}	
//	}	
//	else
//	{
//		key_F_cnt=0;
//		RC_Ctl->key_F_flag=0;
//	}
//	
//	
//		//key_X
//	if(RC_Ctl->key_X==1) 
//	 {
//		key_X_cnt++;
//		if(key_X_cnt== 3)	
//		{
//			RC_Ctl->key_X_flag++;				
//		}	 
//	 }
//	 else
//	 {
//		key_X_cnt=0;
//	 }	
//	
//	
//	//key_G
//	if(RC_Ctl->key_G==1) 
//	{
//		key_G_cnt++;
//		if(key_G_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_G_flag++;
//		}	
//	}	
//	else
//	{
//		key_G_cnt=0;
////		RC_Ctl.key_G_flag=0;
//	}	

//	
//		//key_Q
//	if(RC_Ctl->key_Q==1) 
//	{
//		key_Q_cnt++;
//		if(key_Q_cnt==3)	
//		{
//			RC_Ctl->key_Q_flag++;			
//		}	 
//	}   else
//	 {
//		key_Q_cnt=0;	
//	 }
//	 
//	//key_E
//	if(RC_Ctl->key_E==1) 
//	 {
//		key_E_cnt++;
//		if(key_E_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_E_flag++;	
//		if(RC_Ctl->key_E_flag>1)  RC_Ctl->key_E_flag=0;			
//		}	 
//	 }
//	 else
//	 {
//		key_E_cnt=0;
//		//RC_Ctl.key_E_flag=0;
//	 }	
//	//key_Z
//		if(RC_Ctl->key_Z==1) 
//	{
//		key_Z_cnt++;
//		if(key_Z_cnt==Key_Filter_Num)	//RC_Ctl.key_Z_flag清0的位置可以思考一下
//		{
//			RC_Ctl->key_Z_flag=1;
////		if(RC_Ctl.key_Z_flag>1)	RC_Ctl.key_Z_flag=0;
//		}			
//	}	else
//	 {
//		key_Z_cnt=0;
//		RC_Ctl->key_Z_flag=0;
//	 } 
//	//key_V
//		if(RC_Ctl->key_V==1) 
//	{
//		key_V_cnt++;
//		if(key_V_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_V_flag ++;		
//			if(RC_Ctl->key_V_flag > 1)
//				RC_Ctl->key_V_flag = 0;
//		}			
//	}	else
//	 {
//		key_V_cnt=0;	
//	 } 	 
//	//key_ctrl
//	if(RC_Ctl->key_ctrl==1) 
//	{
//		key_ctrl_cnt++;
//		if(key_ctrl_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_ctrl_flag=1;
//			key_ctrl_cnt=0;	
//		}	
//	} 
//	else
//	{
//		RC_Ctl->key_ctrl_flag=0;
//	}	 
//	//key_shift 
//	 if(RC_Ctl->key_shift==1) 
//	{
//		key_shift_cnt++;
//		if(key_shift_cnt==Key_Filter_Num)	
//		{
//			RC_Ctl->key_shift_flag=1;
//			key_shift_cnt=0;	
//		}	
//	 }	
//	else
//	{
//		RC_Ctl->key_shift_flag=0;
//	}
//	 //mouse_l
//	 if(RC_Ctl->mouse.press_l==1)
//	 {
//		mouse_press_l_cnt++;
//		if(mouse_press_l_cnt==Key_Filter_Num)
//		{
//			RC_Ctl->mouse.press_l_flag=1;
//			mouse_press_l_cnt=0;
//		}
//	 }
//	 else
//	 {
//		 RC_Ctl->mouse.press_l_flag=0;
//	 }
//		 //mouse_r
//		 if(RC_Ctl->mouse.press_r==1)
//		 {
//			mouse_press_r_cnt++;
//			if(mouse_press_r_cnt==Key_Filter_Num)
//			{
//				RC_Ctl->mouse.press_r_flag++;				
//			}
//		 }
//		 else
//		 {
//			// RC_Ctl->mouse.press_r_flag=0;
//			 mouse_press_r_cnt=0;
//		 }
//}