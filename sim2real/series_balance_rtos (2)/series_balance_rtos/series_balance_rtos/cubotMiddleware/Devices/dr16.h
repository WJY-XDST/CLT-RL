//#include "stm32h7xx_hal.h"
//#include "usart.h"

//#ifndef DR16_H
//#define DR16_H

//#include "uart_base.h"



//#define DR16_rxBufferLengh 18	 //< dr16接收缓存区数据长度
//#define Key_Filter_Num 7       //< 按键检测消抖滤波时间(ms)
//#define ACE_SENSE  0.12f //0.05f
//#define ACE_SHACHE 0.009f
//#define PARK_SENSE 0.005f

///**
//  * @brief  详细按键定义，补充接收机结构体内容
//  */
//#define key_W        key[0]
//#define key_A        key[1]
//#define key_S        key[2]
//#define key_D        key[3]
//#define key_shift    key[4]
//#define key_ctrl     key[5]
//#define key_Q        key[6]
//#define key_E        key[7]
//#define key_V        key[8]
//#define key_F        key[9]
//#define key_G        key[10]
//#define key_C        key[11]
//#define key_R        key[12]
//#define key_B        key[13]
//#define key_Z        key[14]
//#define key_X        key[15]
// 
//#define key_W_flag        keyflag[0]
//#define key_A_flag        keyflag[1]
//#define key_S_flag        keyflag[2]
//#define key_D_flag        keyflag[3]
//#define key_shift_flag    keyflag[4]
//#define key_ctrl_flag     keyflag[5]
//#define key_Q_flag        keyflag[6]
//#define key_E_flag        keyflag[7]
//#define key_V_flag        keyflag[8]
//#define key_F_flag        keyflag[9]
//#define key_G_flag        keyflag[10]
//#define key_C_flag        keyflag[11]
//#define key_R_flag        keyflag[12]
//#define key_B_flag        keyflag[13]
//#define key_Z_flag        keyflag[14]
//#define key_X_flag        keyflag[15]

//#define W_Num        0
//#define A_Num        1
//#define S_Num        2
//#define D_Num        3
//#define shift_Num    4
//#define ctrl_Num     5
//#define Q_Num        6
//#define E_Num        7
//#define V_Num        8
//#define F_Num        9
//#define G_Num        10
//#define C_Num        11
//#define R_Num        12
//#define B_Num        13
//#define Z_Num        14
//#define X_Num        15


///**
//  * @brief  接收机接收数据类型, 包含rc遥控器数据、mouse鼠标数据和 keyflag 按键数据
//  */
//typedef struct { 
// struct {       
//	 uint16_t sw;
//	 uint16_t ch0;       
//	 uint16_t ch1;       
//	 uint16_t ch2;      
//	 uint16_t ch3;      
//	 uint8_t  s1; 
//	 uint8_t  s2;	
//	 uint8_t  s1_last; 
//	 uint8_t  s2_last; 
//	}rc; 

// struct {       
//	 int16_t x;       
//	 int16_t y;       
//	 int16_t z;         
//	 uint8_t press_l;     
//	 uint8_t press_r; 
//	 uint8_t press_l_flag;     
//	 uint8_t press_r_flag; 
// }mouse; 

//	struct {     
//		uint16_t v;
//		uint8_t s1ands2;
//		uint8_t keyboard14;
//		uint8_t keyboard15;
//	}keyboard;
//	
//	uint8_t  key[18];   
//	uint8_t  keyflag[18];	
//	uint32_t key_filter_cnt[18];
//	uint8_t  isUnpackaging;  	 //< 解算状态标志位，解算过程中不读取数据
//	uint8_t  isOnline;
//	uint32_t onlineCheckCnt;
//	int16_t OneShoot;
//	int16_t ThreeShoot;
//	uint16_t c;
//	uint16_t d;
//	float Chassis_Y_Integ;//斜坡积分变量
//  float Chassis_X_Integ;
//	int8_t ShootNumber;
//	uint8_t ShootNumberCut;
//	uint8_t Cruise_Mode;
//}RC_Ctrl; 
// 

///**
//  * @brief  dr16数据拆分解算函数
//	* @param[in] rc_ctrl  	为该结构体赋值
//	* @param[in] recBuffer  串口接收缓存区
//	* @param[in] len        缓存区数组数据长度，未使用，仅用来匹配函数类型
//  */


///**
//  * @brief  dr16数据拆分解算函数
//	* @param[in] recBuffer  串口接收缓存区
//	* @param[in] len        缓存区数组数据长度，未使用，仅用来匹配函数类型
//  * @retval    RC_Ctl     接收机数据类型，储存杆量和按键信息
//  */
//uint8_t DR16_callback(uint8_t * recBuffer, uint16_t len);

///**
//  * @brief  初始化接收机数据类型的数据，将杆量和按键信息归零
//	* @param[in] RC_Ctl  接收机数据类型首地址，储存杆量和按键信息
//  * @retval    RC_Ctl  接收机数据类型首地址，储存杆量和按键信息
//  */
//void DR16Init(RC_Ctrl* RC_Ctl);


///**
//  * @brief  按键消抖，检测是否为有效按下
//	* @param[in] RC_Ctl  接收机数据类型首地址
//  * @retval    RC_Ctl  接收机数据类型首地址
//  */
//void PC_keybroad_filter(RC_Ctrl* RC_Ctl);

//void DR16_DataUnpack(RC_Ctrl* rc_ctrl, uint8_t * recBuffer, uint16_t len );
//void DR16_task(void * param);

//extern uint8_t DR16_recData[DR16_rxBufferLengh]__attribute__((at(0x24006000)));
//extern UART_RxBuffer uart1_buffer;
//extern UART_RxBuffer uart3_buffer;
//extern RC_Ctrl rc_Ctrl;


//#endif
