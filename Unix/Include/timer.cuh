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
        void start(std::string message = "") override;
        void end(std::string message = "") override;
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
        void start(std::string message = "") override;
        void end(std::string message = "") override;
};