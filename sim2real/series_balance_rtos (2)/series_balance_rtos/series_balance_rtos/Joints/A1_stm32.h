#ifndef __A132_H_
#define __A132_H_
#include "A1.h"
#include "stm32h7xx.h"
#include "uart_base.h"
#include "usart.h"
//#include "five_link_leg.h"


#define MOTOR_ZERO_ID 0
#define MOTOR_ONE_ID 1
#define MOTOR_TWO_ID 1

#define USART_RXBUF_SIZE 88
#define USART_TXBUF_SIZE 34


void Motor_init(MOTOR_send* pMotor_s,uint8_t ID, UART_HandleTypeDef* huart, float pos);
void Motor_receive_test(UART_HandleTypeDef* huart);
void A1_init();

uint8_t A1_send(UART_HandleTypeDef* huart_x, MOTOR_send* pMotor_s);
uint8_t A1_data_send(UART_HandleTypeDef* huart_x,MOTOR_send* pMotor_s,uint8_t* pBuffer,uint8_t len);
uint8_t A1_data_receive(UART_HandleTypeDef* huart_x,MOTOR_recv* callback_fun(uint8_t, UART_HandleTypeDef*),uint8_t* pBuffer,uint8_t len);
uint8_t A1_fake_callback_left(uint8_t * recBuffer, uint16_t len);
uint8_t A1_fake_callback_right(uint8_t * recBuffer, uint16_t len);


extern UART_RxBuffer uart4_buffer;
extern UART_RxBuffer uart5_buffer;
extern MOTOR_send Motor_zero_send_left;
extern MOTOR_send Motor_one_send_left;
extern MOTOR_send Motor_zero_send_right;
extern MOTOR_send Motor_one_send_right;

//宇树电机数据接收结构体
extern MOTOR_recv Motor_zero_receive_left;
extern MOTOR_recv Motor_one_receive_left;
extern MOTOR_recv Motor_zero_receive_right;
extern MOTOR_recv Motor_one_receive_right;

void Motor_RadPositionMode(MOTOR_send* pMotor_s, float pos);
void Motor_ZeroTorque(MOTOR_send* pMotor_s);
void Motor_VelocityMode(MOTOR_send* pMotor_s, float velocity);
void A1_DampingMode(MOTOR_send* pMotor_s);
void Motor_TorqueMode(MOTOR_send* pMotor_s, float tau);
void A1_Dampingtorque();


//使用中的A1模式封装函数
void A1_init(void);


void A1_cc_Pitch1(MOTOR_send* pMotor_s);
void A1_cc_Pitch2(MOTOR_send* pMotor_s);

void A1_ag_control01(MOTOR_send* pMotor_s);
void A1_ag_control02(MOTOR_send* pMotor_s);

void A1_cc_Pitch1_back(MOTOR_send* pMotor_s);
void A1_cc_Pitch2_back(MOTOR_send* pMotor_s);

void A1_reset_control(MOTOR_send* pMotor_s);




#endif

