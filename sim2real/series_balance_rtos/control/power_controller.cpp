//#include "power_controller.hpp"
//#include "math.h"
//#include "wheel.h"
//static inline bool floatEqual(float a, float b) { return fabs(a - b) < 1e-5f; }
//#define clamp(x, min, max) ((x < min) ? min : (x > max) ? max : x)
//float sumPower = 0.0f;

//namespace Power
//{
//	
//Manager manager(Manager::Motors(&Wheel.Motors3508.motor[0], &Wheel.Motors3508.motor[1]));//没什么用
//Manager::Manager(const Motors &motors_,const float k1_, const float k2_, const float k3_)
//	: motors(motors_),
//	  powerBuff(0.0f),
//      fullBuffSet(0.0f),
//      baseBuffSet(0.0f),
//      fullMaxPower(0.0f),
//      baseMaxPower(0.0f),
//	  powerUpperLimit(0.0f),
//      refereeMaxPower(0.0f),
//      torqueConst(0.0003662),
//	  wConst(0.005453),
//      k1(k1_),
//      k2(k2_),
//      k3(k3_),
//      userConfiguredMaxPower(0.0f)
//{
//	  configASSERT(motors_.motorL != NULL);//验证条件是否成立，若失败则触发错误处理（如打印日志、停机）
//    configASSERT(motors_.motorL != NULL);
//    configASSERT(motors_.motorL != NULL);
//}
//	
//float *getControlledOutput(PowerObj *objs[2])
//{
//	const float k0 = manager.torqueConst;  // torque current rate of the motor, defined as Nm/Output
//	const float kw = manager.wConst;  // torque current rate of the motor, defined as Nm/Output
//	static float newTorqueCurrent[2];//最终输出力矩
//	static float u_yaw;
//	static float u_speed;
//	float cmdPower[2];
//	float sumCmdPower = 0.0f;
//	//float maxPower = clamp(manager.userConfiguredMaxPower, manager.fullMaxPower, manager.baseMaxPower);
//	float maxPower = chassis_power_limit+(buffer_energy-18)*5;
//	for (int i = 0; i < 2; i++)
//	{
//		PowerObj *p = objs[i];
//		cmdPower[i] = p->T_Output * k0 * p->curAv *kw + (p->curAv * kw * p->curAv * kw) * manager.k1 + p->T_Output * k0 * p->T_Output * k0 * manager.k2 +
//                         manager.k3 / (2);
//		sumCmdPower += cmdPower[i];
//		
//	}
//	sumPower = sumCmdPower;
//	if (sumCmdPower > maxPower)//如果超功率
//    {
//		
//			PowerObj *p = objs[0];
//			float k =  objs[0]->U_speed/objs[0]->U_yaw;
//			float A = manager.k2*(2*k*k+2);
//			float B = 2*manager.k2*((k+1)*objs[0]->M+(k-1)*objs[1]->M)+objs[0]->curAv*(k+1)+objs[1]->curAv*(k-1) ;
//			float C = objs[0]->curAv*objs[0]->M + objs[1]->curAv*objs[1]->M +
//				manager.k1*(objs[0]->curAv*objs[0]->curAv+objs[1]->curAv*objs[1]->curAv) +
//				manager.k2*(objs[0]->M*objs[0]->M+objs[1]->M*objs[1]->M)-maxPower;
//			float delta = B*B-4*A*C;
////			if(delta >0.0f)
////			{
////				u_yaw = objs[0]->U_yaw>0? (-B + sqrt(delta))/(2*A)
////											:(-B - sqrt(delta))/(2*A);
////			}
////			else if (floatEqual(delta, 0.0f))  // repeat roots
////			{
////				 u_yaw = -1*B/(2*A);
////											
////			}
////			
////			else 
////			{
////				u_yaw = -1*B/(2*A);
////			}
////             u_speed = k*u_yaw;
//			u_speed =  objs[0]->U_speed*maxPower/sumCmdPower;
//			u_yaw =	 objs[0]->U_yaw*maxPower/sumCmdPower;

//			newTorqueCurrent[0] = (objs[0]->M+u_speed+u_yaw)/k0;
//			newTorqueCurrent[1] = -(objs[1]->M+u_speed-u_yaw)/k0;
//			//newTorqueCurrent[0] = maxPower*(objs[0]->T_Output)/sumCmdPower;
//			//newTorqueCurrent[1] = -maxPower*(objs[1]->T_Output)/sumCmdPower;
//			
//			
//	}
//	else
//	{
//		newTorqueCurrent[0] = objs[0]->T_Output;
//		newTorqueCurrent[1] = -(objs[1]->T_Output);
//	}
//	return newTorqueCurrent;
//}

//void powerDaemon ()//获取功率上限
//{
//	
//}


//void init(const Manager &mana)//初始化
//{
//}
//}
