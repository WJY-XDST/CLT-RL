#include "rl_host.h"
#include <math.h>
#include <string.h>
typedef char RLHost_FloatIs32Bits[(sizeof(float)==4)?1:-1];
static const uint8_t magic[4]={'W','L','R','L'};
static uint16_t payload_size(uint8_t kind) {
    switch(kind) {
    case RL_HOST_HELLO:return 36;
    case RL_HOST_STATUS:return 188;
    case RL_HOST_TORQUE:return 64;
    case RL_HOST_STOP:return 4;
    default:return 0;
    }
}
uint32_t RLHost_GetU32(const uint8_t *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
void RLHost_PutU32(uint8_t *p,uint32_t v) {
    p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);p[2]=(uint8_t)(v>>16);p[3]=(uint8_t)(v>>24);
}
float RLHost_GetF32(const uint8_t *p) {
    uint32_t v=RLHost_GetU32(p);float f;memcpy(&f,&v,4);return f;
}
void RLHost_PutF32(uint8_t *p,float f) {
    uint32_t v;memcpy(&v,&f,4);RLHost_PutU32(p,v);
}
uint32_t RLHost_CRC32(const uint8_t *p,size_t n) {
    uint32_t c=0xffffffffu;size_t i;uint8_t b;
    for(i=0;i<n;++i) {
        c^=p[i];
        for(b=0;b<8;++b)c=(c>>1)^((c&1u)?0xedb88320u:0u);
    }
    return c^0xffffffffu;
}
int RLHost_SeqNewer(uint32_t v,uint32_t previous) {
    uint32_t delta=v-previous;return delta>0u&&delta<0x80000000u;
}
size_t RLHost_Encode(const RLHost_Packet *p,uint8_t *out,size_t capacity) {
    size_t n;
    if(!p||!out||!p->session||!payload_size(p->kind)||p->size!=payload_size(p->kind))return 0;
    n=RL_HOST_HEADER_SIZE+p->size+4u;if(n>capacity)return 0;
    memcpy(out,magic,4);out[4]=RL_HOST_VERSION;out[5]=p->kind;
    out[6]=(uint8_t)p->size;out[7]=(uint8_t)(p->size>>8);
    RLHost_PutU32(out+8,p->session);RLHost_PutU32(out+12,p->sequence);RLHost_PutU32(out+16,p->time_us);
    memcpy(out+20,p->payload,p->size);RLHost_PutU32(out+n-4,RLHost_CRC32(out,n-4));return n;
}
static void discard(RLHost_Parser *p) {
    --p->used;memmove(p->data,p->data+1,p->used);
}
void RLHost_ParserFeed(RLHost_Parser *p,const uint8_t *data,size_t n,RLHost_ReceiveFn callback,void *context) {
    size_t i,total;uint16_t size;RLHost_Packet packet;
    if(!p||(!data&&n))return;
    for(i=0;i<n;++i) {
        if(p->used>=RL_HOST_FRAME_MAX){++p->rejected;p->used=0;}
        p->data[p->used++]=data[i];
        while(p->used) {
            if(memcmp(p->data,magic,p->used<4?p->used:4)!=0){discard(p);continue;}
            if(p->used<RL_HOST_HEADER_SIZE)break;
            size=(uint16_t)((uint16_t)p->data[6]|((uint16_t)p->data[7]<<8));
            if(p->data[4]!=RL_HOST_VERSION||!payload_size(p->data[5])||
               size!=payload_size(p->data[5])||!RLHost_GetU32(p->data+8)) {
                ++p->rejected;discard(p);continue;
            }
            total=RL_HOST_HEADER_SIZE+size+4u;
            if(p->used<total)break;
            if(RLHost_CRC32(p->data,total-4)!=RLHost_GetU32(p->data+total-4)) {
                ++p->rejected;discard(p);continue;
            }
            packet.kind=p->data[5];packet.size=size;packet.session=RLHost_GetU32(p->data+8);
            packet.sequence=RLHost_GetU32(p->data+12);packet.time_us=RLHost_GetU32(p->data+16);
            memcpy(packet.payload,p->data+20,size);
            p->used=(uint16_t)(p->used-total);memmove(p->data,p->data+total,p->used);
            if(callback)callback(context,&packet);
        }
    }
}
static int finite_values(const float *v,size_t n) {
    size_t i;uint32_t bits;
    for(i=0;i<n;++i) {
        memcpy(&bits,&v[i],4);
        if((bits&0x7f800000u)==0x7f800000u)return 0;
    }
    return 1;
}
void RLHost_Disarm(RLHost_Context *c) {
    if(!c)return;
    c->execution_active=0;c->execute_requested=0;c->has_command=0;c->has_inflight=0;c->stop_requested=1;
}
void RLHost_Fault(RLHost_Context *c,RLHost_Error reason) {
    if(!c)return;
    if(!c->fault_latched)c->fault_reason=reason;
    c->fault_latched=1;RLHost_Disarm(c);
}
int RLHost_Init(RLHost_Context *c,const RLHost_Config *cfg) {
    size_t i;uint8_t any_hash=0;
    if(!c||!cfg)return 0;
    memset(c,0,sizeof(*c));c->stop_requested=1;
    for(i=0;i<32;++i)any_hash|=cfg->contract_sha256[i];
    if(!any_hash||!finite_values(cfg->torque_limit_nm,6)||!finite_values(cfg->action_limit,6)||
       !finite_values(cfg->command_min,3)||!finite_values(cfg->command_max,3)||
       !finite_values(&cfg->observation_action_limit,1)||cfg->observation_action_limit<=0||
       cfg->command_timeout_us<1000||cfg->command_timeout_us>1000000||
       cfg->max_source_age_us<1000||cfg->max_source_age_us>1000000) {
        RLHost_Fault(c,RL_HOST_BAD_CONFIG);return 0;
    }
    for(i=0;i<6;++i)if(cfg->torque_limit_nm[i]<=0||cfg->action_limit[i]<=0) {
        RLHost_Fault(c,RL_HOST_BAD_CONFIG);return 0;
    }
    for(i=0;i<3;++i)if(cfg->command_min[i]>cfg->command_max[i]) {
        RLHost_Fault(c,RL_HOST_BAD_CONFIG);return 0;
    }
    c->config=*cfg;c->initialized=1;return 1;
}
void RLHost_SetPermission(RLHost_Context *c,int allowed,int calibrated) {
    if(!c)return;
    c->execution_allowed=(uint8_t)(allowed!=0);c->calibration_verified=(uint8_t)(calibrated!=0);
    if(!allowed||!calibrated)RLHost_Disarm(c);
}
static const RLHost_StateStamp *find_state(const RLHost_Context *c,uint32_t sequence) {
    const RLHost_StateStamp *s=&c->states[sequence%RL_HOST_STATE_HISTORY];
    return (s->valid&&s->sequence==sequence)?s:0;
}
static void receive_packet(void *context,const RLHost_Packet *p) {
    RLHost_Context *c=(RLHost_Context*)context;
    RLHost_Command cmd;const RLHost_StateStamp *s;uint32_t mode;size_t i;
    if(p->kind==RL_HOST_HELLO) {
        mode=RLHost_GetU32(p->payload+32);
        if(memcmp(p->payload,c->config.contract_sha256,32)!=0||mode>1) {++c->rejected_commands;return;}
        if(p->session==c->session&&c->has_rx_sequence&&!RLHost_SeqNewer(p->sequence,c->last_rx_sequence))return;
        RLHost_Disarm(c);c->session=p->session;c->tx_sequence=0;c->last_rx_sequence=p->sequence;
        c->has_rx_sequence=1;c->last_applied_sequence=0;c->execute_requested=(uint8_t)mode;
        memset(c->states,0,sizeof(c->states));memset(c->last_applied_action,0,sizeof(c->last_applied_action));
        return; /* New HELLO cannot clear a latched hardware/protocol fault. */
    }
    if(!c->session||p->session!=c->session||
       (c->has_rx_sequence&&!RLHost_SeqNewer(p->sequence,c->last_rx_sequence))) {++c->rejected_commands;return;}
    if(p->kind==RL_HOST_STOP) {
        c->last_rx_sequence=p->sequence;RLHost_Disarm(c);return;
    }
    if(p->kind!=RL_HOST_TORQUE||c->fault_latched) {++c->rejected_commands;return;}
    s=find_state(c,RLHost_GetU32(p->payload));
    if(!s||(uint32_t)(c->rx_now_us-s->time_us)>c->config.max_source_age_us) {
        RLHost_Fault(c,RL_HOST_STALE_STATE);return;
    }
    if((s->flags&(RL_HOST_SENSOR_VALID|RL_HOST_VELOCITY_VALID))!=
       (RL_HOST_SENSOR_VALID|RL_HOST_VELOCITY_VALID)) {RLHost_Fault(c,RL_HOST_SENSOR_FAULT);return;}
    memset(&cmd,0,sizeof(cmd));cmd.sequence=p->sequence;cmd.source_observation_sequence=s->sequence;
    cmd.source_sample_us=s->time_us;cmd.received_at_us=c->rx_now_us;
    for(i=0;i<6;++i) {
        cmd.torque_nm[i]=RLHost_GetF32(p->payload+4+4*i);
        cmd.effective_action[i]=RLHost_GetF32(p->payload+28+4*i);
        if(!finite_values(&cmd.torque_nm[i],1)||!finite_values(&cmd.effective_action[i],1)||
           fabsf(cmd.torque_nm[i])>c->config.torque_limit_nm[i]+1.e-5f||
           fabsf(cmd.effective_action[i])>c->config.action_limit[i]+1.e-5f) {
            RLHost_Fault(c,RL_HOST_BAD_COMMAND);return;
        }
        if(cmd.torque_nm[i]>c->config.torque_limit_nm[i])cmd.torque_nm[i]=c->config.torque_limit_nm[i];
        if(cmd.torque_nm[i]<-c->config.torque_limit_nm[i])cmd.torque_nm[i]=-c->config.torque_limit_nm[i];
    }
    for(i=0;i<3;++i) {
        cmd.command[i]=RLHost_GetF32(p->payload+52+4*i);
        if(!finite_values(&cmd.command[i],1)||cmd.command[i]<c->config.command_min[i]-1.e-6f||
           cmd.command[i]>c->config.command_max[i]+1.e-6f) {RLHost_Fault(c,RL_HOST_BAD_COMMAND);return;}
    }
    c->last_rx_sequence=p->sequence;c->latest=cmd;c->has_command=1;++c->received_commands;
}
void RLHost_ReceiveBytes(RLHost_Context *c,const uint8_t *data,size_t size,uint32_t now_us) {
    if(!c||!c->initialized)return;
    c->rx_now_us=now_us;RLHost_ParserFeed(&c->parser,data,size,receive_packet,c);
}
void RLHost_Poll(RLHost_Context *c,uint32_t now_us) {
    if(!c||!c->initialized||c->fault_latched)return;
    if(c->execute_requested&&c->has_command) {
        if((uint32_t)(now_us-c->latest.received_at_us)>c->config.command_timeout_us)
            RLHost_Fault(c,RL_HOST_COMMAND_TIMEOUT);
        else if((uint32_t)(now_us-c->latest.source_sample_us)>c->config.max_source_age_us)
            RLHost_Fault(c,RL_HOST_STALE_STATE);
    }
}
size_t RLHost_BuildStatus(RLHost_Context *c,const float obs[27],const float raw[10],
                         uint32_t flags,uint32_t sample_us,uint8_t *out,size_t capacity) {
    RLHost_Packet packet;RLHost_StateStamp *stamp;float gravity_norm,value;size_t i,n;
    if(!c||!c->initialized||!c->session||!obs||!raw||!out||capacity<RL_HOST_STATUS_BYTES)return 0;
    if(!finite_values(obs,27)||!finite_values(raw,10)) {RLHost_Fault(c,RL_HOST_SENSOR_FAULT);return 0;}
    gravity_norm=obs[3]*obs[3]+obs[4]*obs[4]+obs[5]*obs[5];
    if(gravity_norm<.81f||gravity_norm>1.21f) {RLHost_Fault(c,RL_HOST_SENSOR_FAULT);return 0;}
    if(flags&RL_HOST_FAULT)RLHost_Fault(c,RL_HOST_SENSOR_FAULT);
    flags&=RL_HOST_SENSOR_VALID|RL_HOST_VELOCITY_VALID;
    if(c->execution_active&&flags!=(RL_HOST_SENSOR_VALID|RL_HOST_VELOCITY_VALID))
        RLHost_Fault(c,RL_HOST_SENSOR_FAULT);
    if(c->calibration_verified)flags|=RL_HOST_CALIBRATION_VERIFIED;
    if(c->execution_allowed&&c->calibration_verified&&!c->fault_latched)flags|=RL_HOST_EXECUTION_AVAILABLE;
    if(c->execution_active)flags|=RL_HOST_EXECUTION_ACTIVE;
    if(c->fault_latched)flags|=RL_HOST_FAULT;
    memset(&packet,0,sizeof(packet));packet.kind=RL_HOST_STATUS;packet.size=188;
    packet.session=c->session;packet.sequence=c->tx_sequence;packet.time_us=sample_us;
    memcpy(packet.payload,c->config.contract_sha256,32);RLHost_PutU32(packet.payload+32,flags);
    RLHost_PutU32(packet.payload+36,c->last_applied_sequence);
    for(i=0;i<27;++i) {
        value=obs[i];
        if(i>=21) {
            value=c->last_applied_action[i-21];
            if(value>c->config.observation_action_limit)value=c->config.observation_action_limit;
            if(value<-c->config.observation_action_limit)value=-c->config.observation_action_limit;
        }
        RLHost_PutF32(packet.payload+40+4*i,value);
    }
    for(i=0;i<10;++i)RLHost_PutF32(packet.payload+148+4*i,raw[i]);
    n=RLHost_Encode(&packet,out,capacity);
    if(n) {
        stamp=&c->states[c->tx_sequence%RL_HOST_STATE_HISTORY];stamp->sequence=c->tx_sequence;
        stamp->time_us=sample_us;stamp->flags=flags;stamp->valid=1;++c->tx_sequence;
    }
    return n;
}
int RLHost_GetTorques(RLHost_Context *c,uint32_t now_us,RLHost_Command *command) {
    if(!c||!command)return 0;
    RLHost_Poll(c,now_us);
    if(!c->initialized||c->fault_latched||!c->execute_requested||!c->execution_allowed||
       !c->calibration_verified||!c->has_command)return 0;
    c->inflight=c->latest;c->has_inflight=1;*command=c->inflight;c->stop_requested=0;return 1;
}
int RLHost_MarkApplied(RLHost_Context *c,uint32_t sequence,uint32_t now_us) {
    if(!c||!c->has_inflight)return 0;
    /* Application calls Get/Mark in one serialized motor-output section. */
    if(sequence!=c->inflight.sequence) {RLHost_Fault(c,RL_HOST_BAD_ACK);return 0;}
    RLHost_Poll(c,now_us);
    if(c->fault_latched||!c->execute_requested||!c->execution_allowed||!c->calibration_verified)return 0;
    c->last_applied_sequence=sequence;memcpy(c->last_applied_action,c->inflight.effective_action,sizeof(c->last_applied_action));
    c->execution_active=1;return 1;
}

#ifndef RL_HOST_PORTABLE_TEST
#include "usart.h"
#define RX_BYTES 256u
#define RX_QUEUE_BYTES 1024u
#if defined(__CC_ARM) && !defined(__clang__)
__align(32) static uint8_t rx_dma[RX_BYTES];
__align(32) static uint8_t tx_dma[224];
#else
static uint8_t rx_dma[RX_BYTES] __attribute__((aligned(32)));
static uint8_t tx_dma[224] __attribute__((aligned(32)));
#endif
static uint8_t rx_queue[RX_QUEUE_BYTES];
static volatile uint32_t rx_head,rx_tail,rx_overflow,uart_error;
static RLHost_Context *uart_context;
static int dma_accessible(const void *p) {
    uintptr_t a=(uintptr_t)p;
    return (a>=0x24000000u&&a+256u<=0x24080000u)||
           (a>=0x30000000u&&a+256u<=0x30048000u);
}
static int start_receive(void) {
    SCB_CleanInvalidateDCache_by_Addr((uint32_t*)rx_dma,RX_BYTES);
    if(HAL_UARTEx_ReceiveToIdle_DMA(&huart3,rx_dma,RX_BYTES)!=HAL_OK)return 0;
    __HAL_DMA_DISABLE_IT(huart3.hdmarx,DMA_IT_HT);return 1;
}
int RLHost_Uart3Start(RLHost_Context *c) {
    if(!c||!c->initialized||huart3.Instance!=USART3||huart3.Init.BaudRate!=RL_HOST_UART_BAUD||
       !huart3.hdmarx||!huart3.hdmatx||huart3.hdmarx->Init.Mode!=DMA_NORMAL||
       !dma_accessible(rx_dma)||!dma_accessible(tx_dma)||uart_context)return 0;
    rx_head=rx_tail=rx_overflow=uart_error=0;uart_context=c;
    if(!start_receive()){uart_context=0;RLHost_Fault(c,RL_HOST_UART_FAULT);return 0;}
    return 1;
}
void RLHost_Uart3Stop(RLHost_Context *c) {
    if(!c||c!=uart_context)return;
    uart_context=0;__DMB();
    HAL_UART_Abort(&huart3);
    rx_head=rx_tail=rx_overflow=uart_error=0;
    RLHost_Disarm(c);
}
void RLHost_Uart3RxEvent(uint16_t size) {
    uint32_t head,i;
    if(!uart_context)return;
    if(size>RX_BYTES){uart_error=1;return;}
    SCB_InvalidateDCache_by_Addr((uint32_t*)rx_dma,RX_BYTES);
    head=rx_head;
    if(RX_QUEUE_BYTES-(uint32_t)(head-rx_tail)<size)rx_overflow=1;
    else {
        for(i=0;i<size;++i)rx_queue[(head+i)&(RX_QUEUE_BYTES-1u)]=rx_dma[i];
        __DMB();rx_head=head+size;
    }
    if(!start_receive())uart_error=1;
}
void RLHost_Uart3Error(void) {if(uart_context)uart_error=1;}
void RLHost_Uart3Poll(RLHost_Context *c,uint32_t now_us) {
    uint8_t chunk[64];size_t n;uint32_t tail;
    if(!c||c!=uart_context)return;
    if(rx_overflow)RLHost_Fault(c,RL_HOST_RX_OVERFLOW);
    if(uart_error)RLHost_Fault(c,RL_HOST_UART_FAULT);
    do {
        n=0;tail=rx_tail;
        while(tail!=rx_head&&n<sizeof(chunk))chunk[n++]=rx_queue[(tail++)&(RX_QUEUE_BYTES-1u)];
        __DMB();rx_tail=tail;
        if(n)RLHost_ReceiveBytes(c,chunk,n,now_us);
    }while(n);
    RLHost_Poll(c,now_us);
}
int RLHost_Uart3SendStatus(RLHost_Context *c,const float obs[27],const float raw[10],uint32_t flags,uint32_t sample_us) {
    size_t n;
    if(!c||c!=uart_context||huart3.gState!=HAL_UART_STATE_READY)return 0;
    n=RLHost_BuildStatus(c,obs,raw,flags,sample_us,tx_dma,sizeof(tx_dma));
    if(!n)return 0;
    SCB_CleanDCache_by_Addr((uint32_t*)tx_dma,sizeof(tx_dma));
    if(HAL_UART_Transmit_DMA(&huart3,tx_dma,(uint16_t)n)!=HAL_OK){RLHost_Fault(c,RL_HOST_UART_FAULT);return 0;}
    return 1;
}
#endif
