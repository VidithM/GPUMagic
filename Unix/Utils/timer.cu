#include "timer.cuh"

void timer::start(std::string message) {
    if(message.size()){
        std::cout << "[TIMER START] (" << message << ")" << std::endl;
    } else {
        std::cout << "[TIMER START]" << std::endl;
    }
    did_start = true;
    mark = std::chrono::high_resolution_clock::now();
}

void timer::end(std::string message) {
    if(!did_start){
        return;
    }
    std::chrono::time_point<std::chrono::system_clock> end_mark = 
        std::chrono::high_resolution_clock::now();
    
    if(message.size()){
        std::cout 
            << "[TIMER END]: ("
            << message << ") " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_mark - mark).count()
            << " (ms)" << 
        std::endl;
    } else {
        std::cout 
            << "[TIMER END]: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_mark - mark).count()
            << " (ms)" << 
        std::endl;
    }
    did_start = false;
}

void gpu_timer::start(std::string message){
    if(message.size()){
        std::cout << "[GPU TIMER START] (" << message << ")" << std::endl;
    } else {
        std::cout << "[GPU TIMER START]" << std::endl;
    }
    did_start = true;
    cudaEventRecord(start_mark);
    cudaEventSynchronize(start_mark);
}

void gpu_timer::end(std::string message) {
    if(!did_start){
        return;
    }
    cudaEventRecord(end_mark);
    cudaEventSynchronize(end_mark);
    float millis;
    cudaEventElapsedTime(&millis, start_mark, end_mark);
    if(message.size()){
        std::cout << "[GPU TIMER END]: (" << message << ") " << millis << " (ms)" << std::endl;
    } else {
        std::cout << "[GPU TIMER END]: " << millis << " (ms)" << std::endl;
    }
    did_start = false;
}