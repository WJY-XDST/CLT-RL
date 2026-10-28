/**
  *********************************************************************
  * @file      all_init.c/h
  * @brief     该函数用于初始化任务，以及初始化API函数
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */    
/*module*/
#include "uart_base.h"
#include "can_base.h"
#include "motor.h"
#include "all_init.h"
#include "mpu6050.h"
#include "et08.h"
#include "A1_stm32.h"
#include "wheel.h"
#include "vmc.h"
#include "WBR.h"
#include "HI04.h"
#include "DMmotor.h"


/*TASK*/
#include "gyro_task.h"
#include "joint_task.h"
#include "chasis_task.h"
#include "check_task.h"
#include "observe_task.h"
#include "INS_task.h"
#include "stp_task.h"
#include "can_modify_task.h"
#include "online_check_task.h"
#include "LK_motor.h"

/*和freertos相关*/
#include "cmsis_os.h"
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

//static QueueSetHandle_t can_xQueueSet; /* 输入设备的队列集 */
//static QueueHandle_t g_xQueueCan1;/*队列1*/
//static QueueHandle_t g_xQueueCan2;/*队列2*/

/*结构体变量初始化*/
BasePID_Object Turn_PID;
BasePID_Object pid_roll_compen;
BasePID_Object pid_chassis_follow;//底盘跟随pid
BasePID_Object pid_chassis_move;//底盘不lqr控制pid
BasePID_Object pid_legright,pid_legleft;//腿长pid
BasePID_Object pid_lqr_reset;//腿部复位pid
/*函数初始化*/
void MPU_init(void)
{
        //MUP structure variable define
    MPU_Region_InitTypeDef MPU_Config;
    
    /*-----------Open FPU--------*///High speed FLOAT calculate
    SCB->CPACR |= ((3UL << (10*2))|(3UL << (11*2)));  /* set CP10 and CP11 Full Access */
    /*-----------Open Cache------------*/
    SCB_EnableICache();//使能I-Cache
  SCB_EnableDCache();//使能D-Cache   
    SCB->CACR|=1<<2;   //强制D-Cache透写,如不开启,实际使用中可能遇到各种问题    
    /*-----------Open MPU------------*/
    HAL_MPU_Disable();
    MPU_Config.Enable=MPU_REGION_ENABLE;
    MPU_Config.Number=MPU_REGION_NUMBER1;//保护区编号 1
    MPU_Config.BaseAddress= 0x24000000;//保护区基地址
    MPU_Config.Size=MPU_REGION_SIZE_512KB;//设置保护区512k
    MPU_Config.SubRegionDisable=0x00;//禁止子区域
    MPU_Config.TypeExtField=MPU_TEX_LEVEL0;//设置类型扩展域为level0
    MPU_Config.AccessPermission=MPU_REGION_FULL_ACCESS;//全访问（特权&用户都可访问）
    MPU_Config.DisableExec=MPU_INSTRUCTION_ACCESS_ENABLE;//允许指令访问
    MPU_Config.IsShareable=MPU_ACCESS_SHAREABLE;//允许共享
    MPU_Config.IsCacheable=MPU_ACCESS_CACHEABLE;//允许cache
    MPU_Config.IsBufferable=MPU_ACCESS_NOT_BUFFERABLE;//不允许缓冲 DMA模式下要设为不允许
    HAL_MPU_ConfigRegion(&MPU_Config);
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}



void All_init(void)
{
    /*陀螺仪初始化*/
    MPU_init();
    DWT_Init(480);
    INS_Init(&bmi088.bmi088_Data); // 确保BMI088被正确初始化.
    /*将陀螺仪初始化放前面主要为了在陀螺仪初始化之前不打开空闲中断，否则空闲终中断中写消息发送队列而没任务接收导致程序卡死*/
    
    /*初始化串口队列*/
    uart7.g_xQueueUart = xQueueCreate(10, sizeof(uart7_buffer));
    uart8.g_xQueueUart = xQueueCreate(10, sizeof(uart8_buffer));
    uart1.g_xQueueUart = xQueueCreate(10, sizeof(ET08_Rxbuffer));

    /*串口任务*/
    ET08Init(&rc_Ctrl);

    UARTx_Init(&huart1,ET08_callback);
    UART_ENABLE_IT(&uart1,&ET08_Rxbuffer);

    UARTx_Init(&huart7,stp_callback);
    UART_ENABLE_IT(&uart7,&uart7_buffer);

    /*pid初始化*/


    BasePID_Init(&pid_legleft,6000,2,22000,0.03);//5000,15,10000,0.03
    BasePID_Init(&pid_legright,6000,2,22000,0.03);

    //BasePID_Init(&Turn_PID,4,0,20,0);//转向控制PID，不用
    BasePID_Init(&pid_roll_compen, -0.01, 0.00000, 0, 0);//roll补偿

    BasePID_Init(&pid_chassis_follow , 0.9 , 0.0, 80, 0);//平衡底盘跟随
    BasePID_Init(&pid_chassis_move , 1 , 0.001, 0.1, 0);//倒下平衡跟随
    
    /*电机初始化*/
    wheel_joint_Init(&Wheel ,CAN2);
	  LK_MotorInit(&lkmotor[0], 51904,CAN1,1);
	  LK_MotorInit(&lkmotor[1], 37894,CAN1,2);
	  LK_MotorInit(&lkmotor[2], 11792,CAN1,3);
	  LK_MotorInit(&lkmotor[3], 14533,CAN1,4);

    CANx_Init(&hfdcan1,CAN1_rxCallBack);//初始化can回调及can消息队列
    CANx_Init(&hfdcan2,CAN2_rxCallBack);
    CAN_Open(&can1);//打开can
    CAN_Open(&can2);
    
    /*连杆参数初始化*/
    link_length_init(&VMC_joint_left,&VMC_joint_right);
    model_matrix_init();

		UART_Receive_DMA(&uart1, &ET08_Rxbuffer);
		UART_Receive_DMA(&uart7, &uart7_buffer);

    
}
void start_task_freertos(void *params)/*在这创建任务*/
{    
    /*创建处理can数据任务*/
    xTaskCreate(can_data_modify_task, "motordatatask", 160, &can1, osPriorityNormal, NULL);//创建can1数据处理函数
    xTaskCreate(can_data_modify_task, "motordatatask", 160, &can2, osPriorityNormal, NULL);//创建can2数据处理函数

//    xTaskCreate(LK_MotorData_Task, "lkmotordatatask", 128, NULL, osPriorityNormal, NULL);
    xTaskCreate(Gyrodata_task, "gyrodatatask", 256, NULL, osPriorityNormal+1, NULL);//EKF
//    
    /*创建电机最终输出任务*/
    xTaskCreate(joint_task, "jointtask", 192, NULL, osPriorityNormal, NULL);
//    /*创建底盘任务*/
    xTaskCreate(chassis_task, "chassistask", 256, NULL, osPriorityNormal+1, NULL);
    xTaskCreate(addition_task, "additiontask", 128, NULL, osPriorityNormal, NULL);//创建底盘其余功能任务
//    /*创建测距任务*/
    xTaskCreate(STP_task, "stptask", 128, NULL, osPriorityNormal, NULL);//创建测距任务
//    /*模块离线判断任务*/
//    xTaskCreate(online_check_task, "onlinetask", 128, NULL, osPriorityNormal-1, NULL);
//    /*滤波任务*/
    xTaskCreate(Observe_task, "observetask", 128, NULL, osPriorityNormal+1, NULL);
//    /*MRAC*/
//    //xTaskCreate(mrac_task, "mractask", 512*10, NULL, osPriorityNormal+1, NULL);
//    /*遥控器*/
    xTaskCreate(ET08_task, "et08task", 128, NULL, osPriorityNormal+1, NULL);


}