#ifndef _DRIVER_SPI_H_
#define _DRIVER_SPI_H_

#include "gpio.h"
#include "spi.h"
#include <stdlib.h>
#include <stdint.h>

enum SPI_LIST {
    DEVICE_SPI1,
    DEVICE_SPI2,
    DEVICE_SPI3,
    DEVICE_SPI4,
    DEVICE_SPI5,
    DEVICE_SPI6,
    SPI_DEVICE_NUM,
};
/**
 * @brief   SPI接收模式枚举
 */
typedef enum {
    SPI_BLOCK_MODE, // 默认使用阻塞模式
    SPI_IT_MODE,
    SPI_DMA_MODE,
} SPI_TXRX_MODE_e;

/**
 * @brief SPI片选信号对应的GPIO引脚号
 */
typedef struct
{
    GPIO_TypeDef *gpiox; // 片选信号对应的GPIO,如GPIOA,GPIOB等等
    uint16_t cs_pin;     // 片选信号对应的引脚号,GPIO_PIN_1,GPIO_PIN_2等等
} SPI_Chip_Select_t;

/**
 * @brief   SPI接收缓冲区
 */
typedef struct
{
    uint8_t *Data;
    uint16_t Size;
} SPI_RxBuffer_t;

/**
 * @brief	SPI发送缓冲区
 */
typedef struct
{
    uint8_t *Data;
    uint16_t Size;
} SPI_TxBuffer_t;

/**
 * @brief SPI从机结构体
 */
typedef struct _SPI_Slave_t {
    SPI_HandleTypeDef *spiHandler;                // SPI外设handle
    SPI_TXRX_MODE_e spi_work_mode;                // 传输工作模式
    SPI_RxBuffer_t rxBuffer;                      // 接收缓存区结构体
    SPI_TxBuffer_t txBuffer;                      // 发送缓存区结构体
    SPI_Chip_Select_t chipSelect;                 // 片选GPIO引脚结构体
    void (*RxCallBackSPI)(struct _SPI_Slave_t *); // 接收回调函数
} SPI_Slave_t;

/**
 * @brief SPI实例结构体
 */
typedef struct _SPI_Instance_t {
    uint16_t reg_slaves_num;       // 已经注册的片选从机数
    uint16_t total_slaves_num;     // 总的从机数
    SPI_HandleTypeDef *spiHandler; // SPI外设handle
    SPI_Slave_t *slavesInstance;   // SPI从机设备实例结构体指针，用于定义不定长数组
} SPI_Instance_t;

/**
 * @brief   SPI回调函数
 */
typedef void (*SPI_RxCallback)(SPI_Slave_t *);

/**
 * @brief 
 * 
 * @param h_spi 
 * @param slave 
 * @param rxCallback 
 * @return uint8_t 
 */
uint8_t SPI_SlaveInit(SPI_HandleTypeDef *h_spi, SPI_Slave_t *slave, SPI_RxCallback rxCallback);

/**
 * @brief 将SPI从机注册到SPI主设备
 * 
 * @param master 主机SPI端口
 * @param slave 从机设备
 */
void SPI_RegisterSlave(SPI_Instance_t *master, SPI_Slave_t *slave);

/**
 * @brief 设置SPI模式
 *
 * @param spi_ins   SPI从机结构体指针
 * @param spi_mode  工作模式,包括阻塞模式(block),中断模式(IT),DMA模式.详见SPI_TXRX_MODE_e的定义
 */
void SPI_SetMode(SPI_Slave_t *spi_ins, SPI_TXRX_MODE_e spi_mode);

/**
 * @brief 通过spi向对应从机发送数据
 * 
 * @param spi_ins spi实例指针
 * @param ptr_data 要发送的数据
 * @param len 待发送的数据长度
 */
void SPI_Transmit(SPI_Slave_t *spi_ins, uint8_t *ptr_data, uint8_t len);

/**
 * @brief 通过spi从从机获取数据
 * @attention 特别注意:请保证ptr_data在回调函数被调用之前仍然在作用域内,否则析构之后的行为是未定义的!!!
 * 
 * @param spi_ins spi实例指针
 * @param ptr_data 接受数据buffer的首地址
 * @param len 待接收的长度
 */
void SPI_Recv(SPI_Slave_t *spi_ins, uint8_t *ptr_data, uint8_t len);

/**
 * @brief 通过spi利用移位寄存器同时收发数据
 * @todo  后续加入阻塞模式下的timeout参数
 * @attention 特别注意:请保证ptr_data_rx在回调函数被调用之前仍然在作用域内,否则析构之后的行为是未定义的!!!
 * 
 * @param spi_ins spi实例指针
 * @param ptr_data_rx 接收数据地址
 * @param ptr_data_tx 发送数据地址
 * @param len 接收&发送的长度
 */
void SPI_TransRecv(SPI_Slave_t *spi_ins, uint8_t *ptr_data_rx, uint8_t *ptr_data_tx, uint8_t len);

extern SPI_Instance_t spi[6];

#endif
