#include "wheel.h"
#include <et08.h>
#include <math.h>
#include "pid.h"
#include "filter.h"
#include "mpu6050.h"
#include "vmc.h"
#include "WBR.h"
#include "additional_con.h"
#include "all_init.h"
#include "torque.h"

wheel Wheel;


/*3508电机初始化函数*/
void wheel_joint_Init(wheel* Wheel ,CanNumber canx)
{
  MotorInit(&Wheel->Motors3508.motor[0], 0, Motor3508, canx , 0x202);		//左
	MotorInit(&Wheel->Motors3508.motor[1], 0, Motor3508, canx , 0x201);		//右
}	

/*角度累加函数*/
void Angular_accum(wheel* Wheel)
{
			
			//转换为累加角度
			//右
	
			if((Wheel->Motors3508.motor[1].Data.Angle<-100)&&(Wheel->S_joint[1].angle_last>100))
			{
			  Wheel->S_joint[1].angle +=360+Wheel->Motors3508.motor[1].Data.Angle-Wheel->S_joint[1].angle_last;
			}
			else if(( Wheel->Motors3508.motor[1].Data.Angle>100)&&(Wheel->S_joint[1].angle_last<-100))
			{
			  Wheel->S_joint[1].angle +=-360+Wheel->Motors3508.motor[1].Data.Angle-Wheel->S_joint[1].angle_last;
			}
			else 
			{
			  Wheel->S_joint[1].angle +=Wheel->Motors3508.motor[1].Data.Angle-Wheel->S_joint[1].angle_last;
			}
			//左
			if((Wheel->Motors3508.motor[0].Data.Angle<-100)&&(Wheel->S_joint[0].angle_last>100))
			{
			  Wheel->S_joint[0].angle +=360+Wheel->Motors3508.motor[0].Data.Angle-Wheel->S_joint[0].angle_last;
			}
			else if(( Wheel->Motors3508.motor[0].Data.Angle>100)&&(Wheel->S_joint[0].angle_last<-100))
			{
			  Wheel->S_joint[0].angle +=-360+Wheel->Motors3508.motor[0].Data.Angle-Wheel->S_joint[0].angle_last;
			}
			else 
			{
			  Wheel->S_joint[0].angle +=Wheel->Motors3508.motor[0].Data.Angle-Wheel->S_joint[0].angle_last;
			}		
	   Wheel->S_joint[1].angle_last=Wheel->Motors3508.motor[1].Data.Angle;
	   Wheel->S_joint[0].angle_last=Wheel->Motors3508.motor[0].Data.Angle;
	 

			
}



void S_count(wheel* Wheel,RC_Ctrl* rc_ctrl)//里程处理
{
	if(rc_ctrl->isOnline == 1 && groundoff_flag == 0 && jump_flag < 1 && lqr_flag ==1)//初步计算
	{

		Wheel->S_L_dot = -(Wheel->Motors3508.motor[0].Data.SpeedRPM) * 0.006643 * 0.06;//0.06轮半径
		Wheel->S_R_dot = (Wheel->Motors3508.motor[1].Data.SpeedRPM) * 0.006643 * 0.06;
		Wheel->S_dot = (Wheel->S_L_dot + Wheel->S_R_dot) / 2;
		Wheel->S += Wheel->S_dot * 0.002;	   
    }
  else if(rc_ctrl->isOnline == 0 || lqr_flag == 0)//离线清零
  {
      Wheel->S_joint[0].angle=0;
      Wheel->S_joint[1].angle=0;
			Wheel->S_joint[0].angle_last=0;
			Wheel->S_joint[1].angle_last=0;
			Wheel->S_L=0;
      Wheel->S_R=0;
			Wheel->S=0;
      Wheel->S_dot=0;
      Wheel->S_last=0;
	  
	  Wheel->S_joint[1].angle_last = Wheel->Motors3508.motor[1].Data.Angle;
	  Wheel->S_joint[0].angle_last = Wheel->Motors3508.motor[0].Data.Angle;
   }
}

/*离线0力矩*/
void zerotorque_3508()
{
    MotorFillData(&Wheel.Motors3508.motor[0],0);
    MotorFillData(&Wheel.Motors3508.motor[1],0);
}

/*倒地或不立时的底盘控制*/
int target_move=0;//目标速度
void benchchassis_conrol(wheel* Wheel,RC_Ctrl* rc_ctrl)
{
    if(lqr_flag == 0 && fire_flag == 0 && rc_ctrl->isOnline == 1)
	{

		
		if(rc_ctrl->rc.ch1>=1074)
			target_move=(follow_target_com/90)*11*(rc_ctrl->rc.ch1-1024);
		else if(rc_ctrl->rc.ch1<=974)
			target_move=(follow_target_com/90)*11*(rc_ctrl->rc.ch1-1024);
		else
			target_move=0;
		
//		Wheel->Motors3508.motor[0].Data.Output= BasePID_SpeedControl(&pid_chassis_move,target_move+140*(Yaw_motor_angle-follow_target_com),Wheel->Motors3508.motor[0].Data.SpeedRPM);
//		Wheel->Motors3508.motor[1].Data.Output= BasePID_SpeedControl(&pid_chassis_move,-target_move+140*(Yaw_motor_angle-follow_target_com),Wheel->Motors3508.motor[1].Data.SpeedRPM);
		Wheel->Motors3508.motor[0].Data.Output= BasePID_SpeedControl(&pid_chassis_move,(-(rc_ctrl->rc.ch1-1024)-(rc_ctrl->rc.ch2-1024))*2,Wheel->Motors3508.motor[0].Data.SpeedRPM);
		Wheel->Motors3508.motor[1].Data.Output= BasePID_SpeedControl(&pid_chassis_move,(rc_ctrl->rc.ch1-1024-(rc_ctrl->rc.ch2-1024))*2,Wheel->Motors3508.motor[1].Data.SpeedRPM);
		if( Wheel->Motors3508.motor[0].Data.Output>8000)
			  Wheel->Motors3508.motor[0].Data.Output=8000;
		if( Wheel->Motors3508.motor[0].Data.Output<-8000)
			 Wheel->Motors3508.motor[0].Data.Output=-8000;
		if( Wheel->Motors3508.motor[1].Data.Output>8000)
			 Wheel->Motors3508.motor[1].Data.Output=8000;
		if( Wheel->Motors3508.motor[1].Data.Output<-8000)
			 Wheel->Motors3508.motor[1].Data.Output=-8000;

		MotorFillData(&Wheel->Motors3508.motor[0], Wheel->Motors3508.motor[0].Data.Output);
		MotorFillData(&Wheel->Motors3508.motor[1], Wheel->Motors3508.motor[1].Data.Output);

   }

}