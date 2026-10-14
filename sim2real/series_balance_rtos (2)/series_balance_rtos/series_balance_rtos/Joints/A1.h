#ifndef __A1_H_
#define __A1_H_
#include "stm32h7xx.h"

typedef int16_t q15_t;
typedef uint32_t COMData32;

/*************包头***********/
typedef __packed struct {
	// 定义 数据包头
	unsigned char  start[2];     // 包头
	unsigned char  motorID;      // 电机ID  0,1,2   0xBB 表示向所有电机广播（此时无返回）
	unsigned char  reserved;
}COMHead;

/********************************************************************************/
/****************************************发送************************************/
/*****************************************↓↓*************************************/
/************发送数据结构************/
typedef __packed struct {  // 以 4个字节一组排列 ，不然编译器会凑整
	// 定义 数据
	uint8_t  mode;        // 关节模式选择
	uint8_t  ModifyBit;   // 电机控制参数修改位
	uint8_t  ReadBit;     // 电机控制参数发送位
	uint8_t  reserved;

	int32_t  Modify;     // 电机参数修改 的数据 
	//实际给FOC的指令力矩为：
	//K_P*delta_Pos + K_W*delta_W + T
	//						T				W				Pos			Kp			KW
	//位置模式：	0				0				rad*9.1	kp			kw	针对Pos的PD控制
	//速度模式：	0				rad*9.1	0				0				kw	针对W的P控制
	//阻尼模式：	0				0				0				0				kw	速度为0的速度控制
	//力矩模式		t				0				0				0				0		输出恒定力矩，空载时持续加速
	//零力矩模式：0				0				0				0				0		输出轴阻力变小
	q15_t     T;      // 期望关节的输出力矩（电机本身的力矩）x256, 7 + 8 描述
	q15_t     W;      // 期望关节速度 （电机本身的速度） x128,       8 + 7描述	
	int32_t   Pos;      // 期望关节位置 x 16384/6.2832, 14位编码器（主控0点修正，电机关节还是以编码器0点为准）

	q15_t    K_P;      // 关节刚度系数 x2048  4+11 描述
	q15_t    K_W;      // 关节速度系数 x1024  5+10 描述

	uint8_t LowHzMotorCmdIndex;     // 电机低频率控制命令的索引, 0-7, 分别代表LowHzMotorCmd中的8个字节
	uint8_t LowHzMotorCmdByte;      // 电机低频率控制命令的字节

	uint32_t  Res[1];    // 通讯 保留字节  用于实现别的一些通讯内容
	
}MasterComdV3;   // 加上数据包的包头 和CRC 34字节

/************发送协议结构************/
typedef __packed struct {
	// 定义 电机控制命令数据包	
	COMHead       head;    //数据包头

	MasterComdV3  Mdata;   //数据体

	COMData32     CRCdata; //CRC 校验
	
}MasterComdDataV3;//发送数据

/************外部操作发送结构体************/
//Define the message to be send. 【定义发送数据】
typedef struct {
	MasterComdDataV3  motor_send_data;  //The data to be sent to motor. Details are shown in motor_msg.h【电机控制数据结构体，详见motor_msg.h】
	int hex_len;                    //The Bytes count of the message to be sent, it should be 34 for this motor【发送命令字节数, 本电机应为34】
	long long send_time;            //The time that message was sent发送该命令的时间, 微秒(us)
	//The values of motor commands 【待发送的各项数据】
	unsigned short id;              //Motor ID【电机ID】
	unsigned short mode;            //The control mode, 0:free, 5:Open loop slow turning, 10:close loop control【电机控制模式，0:空闲, 5:开环缓慢转动, 10:闭环控制】
	//The following parameters are just motor's parameters, do not concern the reducer. The real torque command to control board is:
	//【以下参数均为电机本身参数，与减速器无关。实际传递给控制板的指令力矩为：】
	//K_P*delta_Pos + K_W*delta_W + T
	float T;                        //Desired output torque of motor【期望电机本身的输出力矩（Nm）】
	float W;                        //Desired output speed of motor【期望电机本身的速度(rad/s)】
	float Pos;                      //Desired shaft position of motor【期望电机本身的位置（rad）】
	float Pos_target;
	float Pos_target_last;
	float delta_x;
	float v_max;
	float vel;
	float acce;
	float d_flag;
	float end_flag;
	float step;
	float K_P;                      //The position stiffness【电机本身的位置刚度系数】
	float K_W;                      //The speed stiffness【电机本身的速度刚度系数】
	float deg_init;
	float pos_LPF;
	
	COMData32 Res;
	  
}MOTOR_send;
/**************************************↑↑****************************************/
/*************************************发送***************************************/
/********************************************************************************/

/********************************************************************************/
/*************************************接收***************************************/
/**************************************↓↓****************************************/
/************接收数据结构体************/
typedef struct {  // 以 4个字节一组排列 ，不然编译器会凑整
	// 定义     数据
	uint8_t     mode;     // 当前关节模式
	uint8_t     ReadBit;  // 电机控制参数修改     是否成功位
	int8_t      Temp;     // 电机当前平均温度   
	uint8_t     MError;   // 电机错误 标识//4
	uint32_t    Read;     // 读取的当前 电机 的控制数据 
	int16_t     T;        // 当前实际电机输出力矩       7 + 8 描述//6
	int16_t     W;        // 当前实际电机速度（高速）   8 + 7 描述
	int32_t     LW;       // 当前实际电机速度（低速）   //6
	int16_t     W2;       // 当前实际关节速度（高速）   8 + 7 描述
	int32_t     LW2;      // 当前实际关节速度（低速）   //6
	int16_t     Acc;      // 电机转子加速度       15+0 描述  惯量较小
	int16_t     OutAcc;   // 输出轴加速度         12+3 描述  惯量较大
	int32_t     Pos;      // 当前电机位置（主控0点修正，电机关节还是以编码器0点为准）
	int32_t     Pos2;     // 关节编码器位置(输出编码器)
	int16_t     gyro[3];  // 电机驱动板6轴传感器数据
	int16_t     acc[3];   
	int16_t     Fgyro[3];  //  
	int16_t     Facc[3];
	int16_t     Fmag[3];
	uint8_t     Ftemp;    // 8位表示的温度  7位（-28~100度）  1位0.5度分辨率
	int16_t     Force16;  // 力传感器高16位数据
	int8_t      Force8;   // 力传感器低8位数据
	uint8_t     FError;   //  足端传感器错误标识
	int8_t      Res[1];   // 通讯 保留字节
	
}ServoComdV3;  // 加上数据包的包头 和CRC 78字节（4+70+4）

/************接收协议结构************/
typedef struct {
	// 定义 电机控制命令数据包	
    COMHead       head;    //数据包头
	
    ServoComdV3  Mdata;   //数据体
	
    COMData32     CRCdata; //CRC 校验
	
}ServoComdDataV3;	//接收数据

/************外部操作接收结构体************/
typedef struct
{
    ServoComdDataV3 motor_recv_data;     //The data received from motor. Details are shown in motor_msg.h【电机接收数据结构体，详见motor_msg.h】
    int hex_len;                    //The Bytes count of the received message, it should be 78 for this motor【接收信息的字节数, 本电机应为78】
    long long resv_time;            //The time of receiving this message, microsecond(us)【接收该命令的时间, 微秒(us)】
    int correct;                    //Whether the received data is correct(1:correct, 0:wrong)【接收数据是否完整（1完整，0不完整）】
    //解读得出的电机数据
    unsigned char motor_id;         //Motor ID【电机ID】
    unsigned char mode;             //The control mode, 0:free, 5:Open loop slow turning, 10:close loop control【电机控制模式，0:空闲, 5:开环缓慢转动, 10:闭环控制】
    int Temp;                       //Temperature【温度】
    unsigned char MError;           //Error code【错误码】

    float T;                        //The output torque of motor【当前实际电机输出力矩】
    float W;                        //The motor shaft speed(without filter)【前实际电机速度（无滤波器）】
    float LW;                       //The motor shaft speed(with filter)【当前实际电机速度（有滤波器）】
    int Acc;                        //The acceleration of motor shaft【电机转子加速度】
    float Pos;                      //The motor shaft position(control board zero fixed)【当前电机位置（主控0点修正，电机关节还是以编码器0点为准）】
		
    float gyro[3];                  //The data of 6 axis inertial sensor on the control board【电机驱动板6轴传感器数据】
    float acc[3];
		
	float angle;
}MOTOR_recv;

/**************************************↑↑****************************************/
/*************************************接收***************************************/
/********************************************************************************/

extern unsigned char check_data_forID(unsigned char* pBuffer); // 数据包校验与ID解析
extern void modify_data(MOTOR_send* pMotor_s,unsigned char* pBuffer);    //压包： 数据Ctype --> 压缩 --> 发送协议包stm32type
extern void extract_data(MOTOR_recv* pMotor_r,unsigned char* pBuffer);  //解包 ： 接收协议包stm32type --> 解压 --> 数据Ctype

////计算CRC校验码，输入为：待校验数据指针，数据包含的4字节整形数量（余数舍去）
////Calculate the CRC. Inputs are: pointer to the data to be calculated, Bytes count/4,(ignore remainder)
//COMData32 crc32_core(COMData32* ptr, COMData32 len); 


#endif

