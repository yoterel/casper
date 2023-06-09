#pragma once

#include <mutex>
#include <condition_variable>
#include <deque>

template <typename T>
class blocking_queue
{
private:
    // size_t                  capacity;
    std::mutex              d_mutex;
    std::condition_variable d_condition;
    std::deque<T>           d_queue;
public:
    // blocking_queue(size_t capacity) : capacity(capacity) {}
    // blocking_queue() : capacity(5) {}
    // ~blocking_queue() {}
    void push(T const& value) {
        {
            std::unique_lock<std::mutex> lock(this->d_mutex);
            d_queue.push_front(value);
        }
        this->d_condition.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lock(this->d_mutex);
        this->d_condition.wait(lock, [=]{ return !this->d_queue.empty(); });
        T rc(std::move(this->d_queue.back()));
        this->d_queue.pop_back();
        return rc;
    }
    bool pop_with_timeout(int timeout_ms, T& value) {
        std::unique_lock<std::mutex> lock(this->d_mutex);
        auto status = this->d_condition.wait_for(lock, std::chrono::milliseconds(timeout_ms), [=]{ return !this->d_queue.empty(); });
        if (!status) {
            return false;
        }
        value = std::move(this->d_queue.back());
        this->d_queue.pop_back();
        return true;
    }
    size_t size() {
        return d_queue.size();
    }
};