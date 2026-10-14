//#include "main.h"
//#include "all_init.h"

//#ifndef USE_POWER_CONTROLLER
//#define USE_POWER_CONTROLLER FALSE


//namespace Power
//{
//	struct Manager
//	{
//			struct Motors  // For overloading
//			{
//				const Motor  *motorL;
//				const Motor  *motorR;
//				const Motor *&operator[](int index)
//				{
//				switch (index)
//				{
//				case 0:
//					return motorL;
//				case 1:
//					return motorR;
//				default:
//					return motorL;
//				}
//			   }
//			
//				Motors(const Motor *motorL_,
//				const Motor *motorR_)
//				: motorL(motorL_),
//				  motorR(motorR_)
//				{
//				}
//		
//			};
//			
//			Manager(
//            const Motors &motors_,
//			const float k1_        = 0.01345f,
//            const float k2_        =  1.41683f,
//            const float k3_        = 0.498f);
//			
//			Motors motors;
//			
//			float powerBuff;
//			float fullBuffSet;
//			float baseBuffSet;
//			float fullMaxPower;
//			float baseMaxPower;
//			float powerUpperLimit;
//			float refereeMaxPower;
//			float userConfiguredMaxPower;
//			
//			float k1;
//			float k2;
//			float k3;	
//			
//			float torqueConst;//力矩常数
//			float wConst;//转速常数
//	};
//	
//	struct PowerObj
//	{
//		public:
//		float T_Output;    	 // torque current command, [-maxOutput, maxOutput], no unit
//		float curAv;         // Measured angular velocity, [-maxAv, maxAv], rad/s
//		float M;         	 // 维持机体平衡的力矩
////		float K;             //yaw与speed系数
//		float U_yaw;         	 // 机体转向的力矩
//		float U_speed;         	 // 机体速度的力矩
//		

//		

//	};
//	float *getControlledOutput(PowerObj *objs[2]);
//}
//extern float sumPower ;
//#endif