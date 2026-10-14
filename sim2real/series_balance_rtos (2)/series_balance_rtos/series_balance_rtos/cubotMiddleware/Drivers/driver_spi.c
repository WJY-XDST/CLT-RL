/**
 **********************************************************************************
 * @file        driver_spi.c
 * @brief       驱动层，面向SPI从机的初始化与配置流程，后续参考HNU的开源对代码进行了修改
 * @details
 * @date        2024-07-26
 * @version     V1.0
 * @copyright   Copyright CUMT-CUBOT (c) 2024
 **********************************************************************************
 * @attention
 * 硬件平台: STM32H750VBT \n
 * SDK版本：-++++
 * @par 修改日志:
 * <table>
 * <tr><th>Date         <th>Version     <th>Author      <th>Description
 * <tr><td>2024-07-26   <td>1.0         <td>EmberLuo    <td>创建初始版本
 * </table>
 *
 **********************************************************************************
 ==============================================================================
                            How to use this driver
 ==============================================================================

    1. Include the "driver_spi.h" header file in your project.
        在你的项目中包含"driver_spi.h"头文件。
    2. Create an instance of SPI_Slave_t for each SPI slave device you want to communicate with.
        为要与之通信的每个SPI从设备创建一个SPI_Slave_t实例。
    3. Initialize each SPI slave using the SPI_Slavelnit() function, providing the necessary parameters.
        使用SPI_Slavelnit()函数初始化每个SPI从属，提供必要的参数，如SPI句柄、从属配置和接收回调函数。
    4. Set the SPI mode for each slave using SPL_SetMode(), choosing between DMA, interrupt, or blocking modes.
        使用SPL_SetMode()设置每个从机的SPI模式，在DMA、中断或阻塞模式之间进行选择。
    5. Use the provided functions to communicate with the SPI slaves:SPI_Transmit() to send data SPI_Recv() to receive data SPI_TransRecv() for fullduplex communication.
        使用提供的函数与SPI从站通信:SPI_Transmit()发送数据SPI_Recv()接收数据SPI_TransRecv()进行全双工通信
    6. Implement the necessary callback functions to handle received data or transmission completion events.
        实现必要的回调函数来处理接收到的数据或传输完成事件。

 **********************************************************************************
 * @attention
 * 硬件平台: STM32H750VBT \n
 * SDK版本：-++++
 * if you had modified this file, please make sure your code does not have many
 * bugs, update the version NO., write dowm your name and the date, the most
 * important is make sure the users will have clear and definite understanding
 * through your new brief.
 **********************************************************************************
 * @exception   SPI_NSS有两种模式，SPI_NSS_Hard和SPI_NSS_Soft。
 *              SPI_NSS_Hard，硬件自动拉高拉低片选，在速率上是远比软件方式控制要高的，
 *              缺点是当STM32为主设备时，同一个SPI上面只能接一个从设备。
 *              SPI_NSS_Soft，软件控制，GPIO控制片选拉高拉低，优点是一个SPI上面可以挂多个设备。
 *              SPI_NSS_Soft也是使用较多的方式, 本驱动暂时只考虑这个模式。
 * @todo        考虑将从机编入链表
 **********************************************************************************
 */

#include "driver_spi.h"

SPI_Instance_t spi[6] = {0};

static uint8_t SPI_SerialNumSelect(SPI_HandleTypeDef *h_spi);

uint8_t SPI_SlaveInit(SPI_HandleTypeDef *h_spi, SPI_Slave_t *slave, SPI_RxCallback rxCallback)
{
    uint8_t spi_case         = SPI_SerialNumSelect(h_spi);
    slave->spiHandler        = h_spi;
    slave->RxCallBackSPI     = rxCallback;
    spi[spi_case].spiHandler = h_spi;
    spi[spi_case].total_slaves_num++;
    if (spi[spi_case].slavesInstance == NULL) {
        spi[spi_case].slavesInstance = (SPI_Slave_t *)malloc(spi[spi_case].total_slaves_num * sizeof(SPI_Slave_t)); // 动态分配内存
    } else {
        spi[spi_case].slavesInstance = (SPI_Slave_t *)realloc(spi[spi_case].slavesInstance, spi[spi_case].total_slaves_num * sizeof(SPI_Slave_t)); // 动态重新分配内存
    }
    if (spi[spi_case].slavesInstance == NULL) {
        return 1; // 内存分配失败
    }
    SPI_RegisterSlave(&spi[spi_case], slave);
    return 0;
}

/**
 * @brief 将SPI从机注册到SPI主设备
 *
 * @param master    SPI实例结构体
 * @param slave     SPI从机结构体
 */
void SPI_RegisterSlave(SPI_Instance_t *master, SPI_Slave_t *slave)
{
    // 检查该从机是否已经注册过
    for (uint16_t i = 0; i < master->reg_slaves_num; i++) {
        if (&master->slavesInstance[i] == slave) {
            // 该从机已经注册过，更新其注册信息
            master->slavesInstance[i] = *slave;
            return;
        }
    }
    // 检查是否还有未注册的从机
    if (master->reg_slaves_num >= master->total_slaves_num) {
        return; // 已经没有未注册的从机
    }
    // 将从机结构体复制到主设备SPI实例结构体中
    master->slavesInstance[master->reg_slaves_num] = *slave;
    // 更新已注册的从机数
    master->reg_slaves_num++;
}

/**
 * @brief 设置SPI模式
 *
 * @param spi_ins   SPI从机结构体
 * @param spi_mode  SPI模式枚举
 */
void SPI_SetMode(SPI_Slave_t *spi_ins, SPI_TXRX_MODE_e spi_mode)
{
    if (spi_mode != SPI_DMA_MODE && spi_mode != SPI_IT_MODE && spi_mode != SPI_BLOCK_MODE)
        while (1); // error mode! 请查看是否正确设置模式，或出现指针越界导致模式被异常修改的情况
    spi_ins->spi_work_mode = spi_mode;
}

void SPI_Transmit(SPI_Slave_t *spi_ins, uint8_t *ptr_data, uint8_t len)
{
    // 拉低片选,开始传输(选中从机)
    HAL_GPIO_WritePin(spi_ins->chipSelect.gpiox, spi_ins->chipSelect.cs_pin, GPIO_PIN_RESET);
    switch (spi_ins->spi_work_mode) {
        case SPI_DMA_MODE:
            HAL_SPI_Transmit_DMA(spi_ins->spiHandler, ptr_data, len);
            break;
        case SPI_IT_MODE:
            HAL_SPI_Transmit_IT(spi_ins->spiHandler, ptr_data, len);
            break;
        case SPI_BLOCK_MODE:
            HAL_SPI_Transmit(spi_ins->spiHandler, ptr_data, len, 1000); // 默认50ms超时
            // 阻塞模式不会调用回调函数,传输完成后直接拉高片选结束
            HAL_GPIO_WritePin(spi_ins->chipSelect.gpiox, spi_ins->chipSelect.cs_pin, GPIO_PIN_SET);
            break;
        default:
            while (1); // error mode! 请查看是否正确设置模式，或出现指针越界导致模式被异常修改的情况
    }
}

void SPI_Recv(SPI_Slave_t *spi_ins, uint8_t *ptr_data, uint8_t len)
{
    // 用于稍后回调使用
    spi_ins->rxBuffer.Size = len;
    spi_ins->rxBuffer.Data = ptr_data;
    // 拉低片选,开始传输
    HAL_GPIO_WritePin(spi_ins->chipSelect.gpiox, spi_ins->chipSelect.cs_pin, GPIO_PIN_RESET);
    switch (spi_ins->spi_work_mode) {
        case SPI_DMA_MODE:
            HAL_SPI_Receive_DMA(spi_ins->spiHandler, ptr_data, len);
            break;
        case SPI_IT_MODE:
            HAL_SPI_Receive_IT(spi_ins->spiHandler, ptr_data, len);
            break;
        case SPI_BLOCK_MODE:
            HAL_SPI_Receive(spi_ins->spiHandler, ptr_data, len, 1000);
            // 阻塞模式不会调用回调函数,传输完成后直接拉高片选结束
            HAL_GPIO_WritePin(spi_ins->chipSelect.gpiox, spi_ins->chipSelect.cs_pin, GPIO_PIN_SET);
            break;
        default:
            while (1); // error mode! 请查看是否正确设置模式，或出现指针越界导致模式被异常修改的情况
    }
}

void SPI_TransRecv(SPI_Slave_t *spi_ins, uint8_t *ptr_data_rx, uint8_t *ptr_data_tx, uint8_t len)
{
    // 用于稍后回调使用,请保证ptr_data_rx在回调函数被调用之前仍然在作用域内,否则析构之后的行为是未定义的!!!
    spi_ins->rxBuffer.Size = len;
    spi_ins->rxBuffer.Data = ptr_data_rx;
    // 拉低片选,开始传输
    HAL_GPIO_WritePin(spi_ins->chipSelect.gpiox, spi_ins->chipSelect.cs_pin, GPIO_PIN_RESET);
    switch (spi_ins->spi_work_mode) {
        case SPI_DMA_MODE:
            HAL_SPI_TransmitReceive_DMA(spi_ins->spiHandler, ptr_data_tx, ptr_data_rx, len);
            break;
        case SPI_IT_MODE:
            HAL_SPI_TransmitReceive_IT(spi_ins->spiHandler, ptr_data_tx, ptr_data_rx, len);
            break;
        case SPI_BLOCK_MODE:
            HAL_SPI_TransmitReceive(spi_ins->spiHandler, ptr_data_tx, ptr_data_rx, len, 1000); // 默认50ms超时
            // 阻塞模式不会调用回调函数,传输完成后直接拉高片选结束
            HAL_GPIO_WritePin(spi_ins->chipSelect.gpiox, spi_ins->chipSelect.cs_pin, GPIO_PIN_SET);
            break;
        default:
            while (1); // error mode! 请查看是否正确设置模式，或出现指针越界导致模式被异常修改的情况
    }
}

/**
 * @brief 当SPI接收完成,将会调用此回调函数,可以进行协议解析或其他必须的数据处理等
 *
 * @param h_spi spi handle
 */
void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *h_spi)
{
    uint8_t spi_case = SPI_SerialNumSelect(h_spi);
    for (size_t i = 0; i < spi[spi_case].total_slaves_num; i++) {
        // 如果是当前spi硬件发出的complete,且cs_pin为低电平(说明正在传输),则尝试调用回调函数
        if (HAL_GPIO_ReadPin(spi[spi_case].slavesInstance[i].chipSelect.gpiox, spi[spi_case].slavesInstance[i].chipSelect.cs_pin) == GPIO_PIN_RESET) {
            // 先拉高片选,结束传输,在判断是否有回调函数,如果有则调用回调函数
            HAL_GPIO_WritePin(spi[spi_case].slavesInstance[i].chipSelect.gpiox, spi[spi_case].slavesInstance[i].chipSelect.cs_pin, GPIO_PIN_SET);
            // @todo 后续添加holdon模式,由用户自行决定何时释放片选,允许进行连续传输
            if (spi[spi_case].slavesInstance[i].RxCallBackSPI != NULL) // 回调函数不为空, 则调用回调函数
                spi[spi_case].slavesInstance[i].RxCallBackSPI(&spi[spi_case].slavesInstance[i]);
            return;
        }
    }
}

/**
 * @brief   这是对HAL库的__weak函数的重写,传输使用IT或DMA模式,在传输完成时会调用此函数。
 * @details 在全双工模式下使用 HAL_SPI_TransmitReceive_IT()或HAL_SPI_TransmitReceive_DMA()函数启动SPI传输时,
 *          应该实现HAL_SPI_TxRxCpltCallback()来处理传输完成事件, 因为在这种情况下, HAL_SPI_RxCpltCallback()不会被调用。
 *          和RxCpltCallback共用解析即可,这里只是形式上封装一下,不用重复写。
 * @param h_spi spi handle
 */
void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *h_spi)
{
    HAL_SPI_RxCpltCallback(h_spi); // 直接调用接收完成的回调函数
}

/**
 * @brief 通过句柄返回SPI编号
 *
 * @param h_spi
 * @return uint8_t
 */
uint8_t SPI_SerialNumSelect(SPI_HandleTypeDef *h_spi)
{
    uint8_t spi_case = 0;
    if (h_spi->Instance == SPI1) {
        spi_case = DEVICE_SPI1;
    } else if (h_spi->Instance == SPI2) {
        spi_case = DEVICE_SPI2;
    } else if (h_spi->Instance == SPI3) {
        spi_case = DEVICE_SPI3;
    } else if (h_spi->Instance == SPI4) {
        spi_case = DEVICE_SPI4;
    } else if (h_spi->Instance == SPI5) {
        spi_case = DEVICE_SPI5;
    } else if (h_spi->Instance == SPI6) {
        spi_case = DEVICE_SPI6;
    }
    return spi_case;
}
