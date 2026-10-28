/**
  *********************************************************************
  * @file      chassis_task.c/h
  * @brief     该任务用于底盘解算，vmc，矩阵运算，逆解，状态控制等，执行频率2ms
  * @note       
  * @history
  *
  @verbatim
  ==============================================================================

  ==============================================================================
  @endverbatim
  *********************************************************************
  */
	
#include "motor.h"
#include "can_base.h"
#include "all_init.h"
#include "driver_counter.h"
#include "torque.h"
#include "A1_stm32.h"
#include "WBR.h"
#include "wheel.h"
#include "torque.h"
#include "ins.h"
#include "filter.h"
#include "observe_task.h"
#include "additional_con.h"
#include "power_con.h"
#include "online_check_task.h"
/*freertos*/
#include "freertos.h"
#include "FreeRTOS.h"
#include "task.h"
UBaseType_t uxHighWaterMark_chassis;
int chassis_count;//测试执行频率


void chassis_task(void *param)
{
	TickType_t xLastWakeTime;
	xLastWakeTime = xTaskGetTickCount();
	chassis_count = 0;
	while(1)
	{
		chassis_count++;	
		/*board_interaction*/
		//  board_conmmunity();
		/*vmc_cacl*/
		VMC_Count(&VMC_joint_left,&VMC_joint_right);
		/*s_calc*/
		S_count(&Wheel,&rc_Ctrl);//里程处理
		/*state_control*/
		state_control();
		/*model_calc*/
		model_state();
		
		/*F0_com*/
		F0_control(&VMC_joint_left);
		F0_control(&VMC_joint_right);
		
		/*pid turn control*/
		
		/*jump control*/

		jump_control(&VMC_joint_left,&VMC_joint_right);
		change_leg_gostep();
		/*F_calc*/
		F_Calc(&VMC_joint_left);
		F_Calc(&VMC_joint_right);
		/*no balance_follow*/
//		benchchassis_conrol(&Wheel,&rc_Ctrl);
		 
		/*print*/
//		Usart2DmaPrintf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",VMC_joint_left.theta0,VMC_joint_right.theta0,VMC_joint_left.torque_b,VMC_joint_left.torque_f,VMC_joint_right.torque_b,VMC_joint_right.torque_f);
		Usart2DmaPrintf("%f,%f,%f,%f\n",spring_feedforwardl,spring_feedforwardr,VMC_joint_left.Fn,VMC_joint_right.Fn);
//		Usart2DmaPrintf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",Wheel.S,Wheel.S_dot,model_target.s,model_target.s_dot,distance_target,LPF_XS.last_output);
//		Usart2DmaPrintf("%.4f,%.4f,%d,%d,%d,%d,%d\n",VMC_joint_left.theta0,VMC_joint_right.theta0,lqr_flag,lqr_ready_flag,leg_flag,leg_right_ready,leg_left_ready);
//	Usart2DmaPrintf("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",state[0],state[1],state[4],state[5],state[8],state[9],VMC_joint_left.TW,VMC_joint_right.TW);
//  Usart2DmaPrintf("%f,%f,%f,%f,%f,%f\n",Real_real_x,Real_real_xdot,model_target.s,model_target.s_dot,Wheel.S,Wheel.S_dot);

//		Usart2DmaPrintf("%.4f,%.4f,%.4f\n",length_target,VMC_joint_left.L,VMC_joint_right.L);
        //Usart2DmaPrintf("%f,%d,%d\n",distance_target,groundoff_flag,landing_flag);
		uxHighWaterMark_chassis = uxTaskGetStackHighWaterMark(NULL);
		/*delay*/
		vTaskDelayUntil(&xLastWakeTime,2);//2ms
	}

}
