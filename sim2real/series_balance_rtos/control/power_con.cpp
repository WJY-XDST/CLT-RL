//#include "can_base.h"
//#include "motor.h"
//#include "power_con.h"
//#include "freertos.h"
//#include "all_init.h"
//#include "power_controller.hpp"
//#include "wheel.h"
//#include "vmc.h"
//float sum_power;
//int count_motor;
//float test_out[2];
//void power_control(wheel *Wheel)
//{
//	Power::PowerObj objs[2];
//    Power::PowerObj *pObjs[2] = {&objs[0], &objs[1]};

//		objs[0].curAv=(Wheel->Motors3508.motor[0].Data.SpeedRPM);
//		objs[0].T_Output=(Wheel->Motors3508.motor[0].Data.Output);
//		objs[1].curAv=-(Wheel->Motors3508.motor[1].Data.SpeedRPM);
//		objs[1].T_Output=-(Wheel->Motors3508.motor[1].Data.Output);
//		objs[0].U_speed = VMC_joint_left.T_s;
//		objs[0].U_yaw = VMC_joint_left.T_yaw;
//		objs[0].M = VMC_joint_left.T_balance;
//		objs[1].M = VMC_joint_right.T_balance;
//		
//			
//		float *controlledOutput=Power::getControlledOutput(pObjs);
//		sum_power = sumPower;
//		for (int i = 0; i < 2; i++)
//		{
//			//Wheel->Motors3508.motor[i].Data.Output = controlledOutput[i];
//			
//			//test_out[i] = controlledOutput[i];
//		}
//		
////	}

//}



