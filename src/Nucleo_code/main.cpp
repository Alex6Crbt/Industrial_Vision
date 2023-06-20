/*
#include "mbed.h"
#define T_CLK 3


DigitalIn capt(D4);
PwmOut myservo(D3);
DigitalOut half (D6);
DigitalOut CW (D10);
PwmOut  CLK(D11);
DigitalOut Enable (D9);


 int main() { 
    
    half = 0;
    CW = 1;
    CLK.period_ms(T_CLK);
    CLK.write(0.5); 
     
     while (1){
        wait_us(10);
        Enable =1 ;
     if (capt==1){
        myservo.period_ms(20);


         myservo.pulsewidth_us(2500);
            wait_us(1500000);
         myservo.pulsewidth_us(1000);
         wait_us (200000);
     }
     }
     }


#include "mbed.h"
PwmOut myservo(D3);
DigitalIn capt(D2);

 int main() {  
     while (1){
     myservo.period_ms(20);
     myservo.pulsewidth_us(1000);
     if (capt==1){
        
            myservo.pulsewidth_us(2500);
            wait(1.5);
        
            myservo.pulsewidth_us(1000);
            wait(1.5);
         }     
     }
     }


*/// Code moteur tapis

#include "mbed.h"
#define T_CLK 3


UnbufferedSerial      my_pc(USBTX, USBRX);

char data_piston;

void ISR_my_pc_reception(void);

DigitalOut half (D6);
DigitalOut CW (D7);
PwmOut  CLK(D11);
DigitalOut Enable (D4);


PwmOut myservo1(D3);
PwmOut myservo2(D9);
PwmOut myservo3(D5);

DigitalIn capt(D2);

int main(){
  half = 0;
  CW = 1;
  CLK.period_ms(T_CLK);
  CLK.write(0.5); 
  Enable =1 ;
  {    
    myservo1.period_ms(20);
    myservo2.period_ms(20);
    myservo3.period_ms(20);
    my_pc.baud(115200);
    my_pc.attach(&ISR_my_pc_reception, UnbufferedSerial::RxIrq);
    while (true){}
}

}


void ISR_my_pc_reception(void){
    my_pc.read(&data_piston, 1);     // get the received byte
   


    if(data_piston == '1'){ // echo of the byte received
        if (capt==1){
            wait_us(130000);
        myservo1.pulsewidth_us(2500);
            wait_us (1500000);
        myservo1.pulsewidth_us(1000);
    }
    }
    


    if(data_piston == '2'){  
        if (capt==1){
            wait_us(100000);  
        myservo2.pulsewidth_us(2500);
            wait_us (1500000);
        myservo2.pulsewidth_us(1000);
    }
    }

    if(data_piston == '3'){   
        if (capt==1){
           wait_us(1000000);
        myservo3.pulsewidth_us(2500);
            wait_us (1500000);
        myservo3.pulsewidth_us(1000);     

        }
    }
     }