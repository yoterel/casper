#include <chrono>
#include <thread>
#include "queue.h"
#include "camera.h"
#include "display.h"
#include "SerialPort.h"

void testing() {
    blocking_queue<int> queue;
    // std::thread flush(FlushQueue, std::ref(q), &num_elems);
    // create producers
    std::vector<std::thread> producers;
    for (int i = 0; i < 10; i++) {
        producers.push_back(std::thread([&queue, i]() {
            queue.push(i);
            std::cout << "Produced: " << i << std::endl;
            // produces too fast
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }));
    }

    // create consumers
    std::vector<std::thread> consumers;
    for (int i = 0; i < 10; i++) {
        producers.push_back(std::thread([&queue, i]() {
            int i = queue.pop();
            std::cout << "Consumed: " << i << std::endl;
            // consumes too slowly
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }));
    }

    std::for_each(producers.begin(), producers.end(), [](std::thread &thread) {
        thread.join();
    });

    std::for_each(consumers.begin(), consumers.end(), [](std::thread &thread) {
        thread.join();
    });
}