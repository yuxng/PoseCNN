#pragma once

#include <iostream>
#include <map>
#include <string>

#include <pangolin/utils/timer.h>

namespace df {

// GlobalTimer is a Myers singleton as described in Modern C++ Design 6.4
// It therefore has the Dead Reference problem (described in  6.5), but
// there are no dependencies so this should be OK.
class GlobalTimer {
public:

    static GlobalTimer & getTimer() {
        static GlobalTimer timer;
        return timer;
    }

    static inline void tick(const std::string name) {
        getTimer().tick_(name);
    }

    static inline void tock(const std::string name) {
        getTimer().tock_(name);
    }

private:
    GlobalTimer() { }
    GlobalTimer(const GlobalTimer &);
    GlobalTimer & operator=(const GlobalTimer &);

    inline void tick_(const std::string name) {
        if (timings_.find(name) == timings_.end()) {
            timings_[name] = { 0.0, 0, pangolin::TimeNow() };
        } else {
            timings_[name].start_ = pangolin::TimeNow();
        }
    }

    inline void tock_(const std::string name) {
        pangolin::basetime end = pangolin::TimeNow();
        ClockInfo & clock = timings_[name];
        const double timeDiff = pangolin::TimeDiff_us(clock.start_,end);
        ++clock.count_;
        clock.totalTime_ += timeDiff;
    }

    ~GlobalTimer() {
        for (std::pair<const std::string,ClockInfo> & timing : timings_) {
            const double averageTime = timing.second.totalTime_ / timing.second.count_;
            std::cout << timing.first << ": " << (averageTime*0.001) << " ms on average (" << timing.second.count_ << " calls)" << std::endl;
        }
    }

    struct ClockInfo {
        double totalTime_;
        int count_;
        pangolin::basetime start_;
    };

    std::map<std::string,ClockInfo> timings_;

};

} // namespace df
