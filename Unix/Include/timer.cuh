#pragma once

#include "common.cuh"
#include <chrono>

class timer_base {
    public:
        virtual void start(std::string message = "") = 0;
        virtual void end(std::string message = "") = 0;
};

class timer : public timer_base {
    private:
        std::chrono::time_point<std::chrono::system_clock> mark;
        bool did_start;
    public:
        timer() : did_start(false) {}
        void start(std::string message = "") override {
            std::cout << "[TIMER START] (" << message << ")" << std::endl;
            did_start = true;
            mark = std::chrono::high_resolution_clock::now();
        }
        void end(std::string message = "") override {
            if(!did_start){
                return;
            }
            std::chrono::time_point<std::chrono::system_clock> end_mark = 
                std::chrono::high_resolution_clock::now();

            std::cout 
                << "[TIMER END]: ("
                << message << ") " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_mark - mark).count()
                << " (ms)" << 
            std::endl;
            did_start = false;
        }
};

class gpu_timer : public timer_base {
    private:
        cudaEvent_t start_mark, end_mark;
        bool did_start;
    public:
        gpu_timer() : did_start(false) {
            cudaEventCreate(&start_mark);
            cudaEventCreate(&end_mark);
        }
        ~gpu_timer(){
            cudaEventDestroy(start_mark);
            cudaEventDestroy(end_mark);
        }
        void start(std::string message = "") override {
            std::cout << "[GPU TIMER START] (" << message << ")" << std::endl;
            did_start = true;
            cudaEventRecord(start_mark);
            cudaEventSynchronize(start_mark);
        }
        void end(std::string message = "") override {
            if(!did_start){
                return;
            }
            cudaEventRecord(end_mark);
            cudaEventSynchronize(end_mark);
            float millis;
            cudaEventElapsedTime(&millis, start_mark, end_mark);
            std::cout << "[GPU TIMER END]: (" << message << ") " << millis << " (ms)" << std::endl;
            did_start = false;
        }
};