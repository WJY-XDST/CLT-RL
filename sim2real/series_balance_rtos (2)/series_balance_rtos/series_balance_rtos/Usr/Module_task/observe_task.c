/**
  *********************************************************************
  * @file      observe_task.c/h
  * @brief     该任务是对机体运动速度估计，用于抑制打滑
	* 					 原理来源于https://zhuanlan.zhihu.com/p/689921165
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */
	
#include "observe_task.h"
#include "kalman_filter.h"
#include "cmsis_os.h"
#include "wheel.h"
#include "ins.h"
#include "vmc.h"
#include "all_init.h"
#include "vmc.h"
#include "WBR.h"
#include "additional_con.h"
#include "all_init.h"
#include "et08.h"
KalmanFilter_t vaEstimateKF;	   // 卡尔曼滤波器结构体

float vaEstimateKF_F[4] = {1.0f, 0.002f, 
                           0.0f, 1.0f};	   // 状态转移矩阵，控制周期为0.002s

float vaEstimateKF_P[4] = {0.0f, 0.0f,
                           0.0f, 0.0f};    // 后验估计协方差初始值

float vaEstimateKF_Q[4] = {2.0f, 0.0f, 
                           0.0f, 4.0f};    // Q矩阵初始值

float vaEstimateKF_R[4] = {500.0f, 0.0f, 
                            0.0f,  1000.0f}; 	
														
float vaEstimateKF_K[4];
													 
const float vaEstimateKF_H[4] = {1.0f, 0.0f,
                                 0.0f, 1.0f};	// 设置矩阵H为常量
														 															 
int observe_count = 0;//任务计数				 
uint32_t OBSERVE_TIME=2;//任务周期是2ms	
float vel_acc[2];//滤波值 							 
float Real_real_x = 0;//滤波位移
float Real_real_xdot = 0;//滤波速度
								 
UBaseType_t uxHighWaterMark_observe;//测试剩余栈大小			
								 
void Observe_task(void *param)
{
	static float wr,wl=0.0f;
	static float vrb,vlb=0.0f;
	float aver_v=0.0f;		
	float acc_y	=0;		
	xvEstimateKF_Init(&vaEstimateKF);
	while(1)
	{  
		observe_count ++;
		
		if(observe_count > 1000 && (rc_Ctrl.isOnline == 1 && groundoff_flag == 0 && jump_flag < 1 && lqr_flag == 1))
		{
		wr= Wheel.S_R_dot/0.06f+INS_attitude->gyro[0] + VMC_joint_right.u0_dot;//右边驱动轮转子相对大地角速度，这里定义的是顺时针为正
		vrb=wr*0.06+VMC_joint_right.L*VMC_joint_right.theta0_speed*arm_cos_f32(VMC_joint_right.theta0)+VMC_joint_right.L_dot*arm_sin_f32(VMC_joint_right.theta0);//机体b系的速度
		
		wl=  Wheel.S_L_dot/0.06f+INS_attitude->gyro[0] + VMC_joint_left.u0_dot;////左边驱动轮转子相对大地角速度，这里定义的是顺时针为正
		vlb=wl*0.06+VMC_joint_left.L*VMC_joint_left.theta0_speed*arm_cos_f32(VMC_joint_left.theta0)+VMC_joint_left.L_dot*arm_sin_f32(VMC_joint_left.theta0);//机体b系的速度
		
		aver_v=1.0f*(vrb+vlb)/2.0f;//取平均
		acc_y=((INS_attitude->accel[1]) * cosf((INS_attitude->pitch ) * AtR) - (INS_attitude->accel[2]) * sinf((INS_attitude->pitch) * AtR)) ;	
        if ((acc_y < 0.1) && (acc_y > -0.1))
			acc_y = 0;
		xvEstimateKF_Update(&vaEstimateKF,acc_y,aver_v);
		
		//原地自转的过程中v_filter和x_filter应该都是为0
		Real_real_xdot=vel_acc[0];//得到卡尔曼滤波后的速度
		Real_real_x=Real_real_x+Real_real_xdot*((float)OBSERVE_TIME/1000.0f);
		}
		else if(rc_Ctrl.isOnline==0 || lqr_flag == 0)//离线清零
		{
			acc_y = 0;
			aver_v = 0;
			Real_real_x = 0;
			Real_real_xdot = 0;
		}
		vTaskDelay(OBSERVE_TIME);
		uxHighWaterMark_observe = uxTaskGetStackHighWaterMark(NULL );
	}
}
	

void xvEstimateKF_Init(KalmanFilter_t *EstimateKF)
{
    Kalman_Filter_Init(EstimateKF, 2, 0, 2);	// 状态向量2维 没有控制量 测量向量2维
	
	memcpy(vaEstimateKF.F_data, vaEstimateKF_F, sizeof(vaEstimateKF_F));
    memcpy(vaEstimateKF.P_data, vaEstimateKF_P, sizeof(vaEstimateKF_P));
    memcpy(vaEstimateKF.Q_data, vaEstimateKF_Q, sizeof(vaEstimateKF_Q));
    memcpy(vaEstimateKF.R_data, vaEstimateKF_R, sizeof(vaEstimateKF_R));
    memcpy(vaEstimateKF.H_data, vaEstimateKF_H, sizeof(vaEstimateKF_H));

}

void xvEstimateKF_Update(KalmanFilter_t *EstimateKF ,float acc,float vel)
{   	
    //卡尔曼滤波器测量值更新
    EstimateKF->MeasuredVector[0] =	vel;//测量速度
    EstimateKF->MeasuredVector[1] = acc;//测量加速度
    		
    //卡尔曼滤波器更新函数
    Kalman_Filter_Update(EstimateKF);

    // 提取估计值
    for (uint8_t i = 0; i < 2; i++)
    {
      vel_acc[i] = EstimateKF->FilteredValue[i];
    }
}


