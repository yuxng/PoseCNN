#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/utils/timer.h>

namespace df {

class FPSCounter {
public:

    explicit FPSCounter(const std::string varName,
                        const int displayFrequency = 10)
        : fps_(varName,0),
          displayFrequency_(displayFrequency),
          i_(0),
          lastTime_(pangolin::TimeNow()) { }

    inline void update() {
        ++i_;
        if (i_ == displayFrequency_) {
            pangolin::basetime time = pangolin::TimeNow();
            const double secondsElapsed = pangolin::TimeDiff_s(lastTime_,time);
            fps_ = displayFrequency_ / secondsElapsed;
            lastTime_ = time;
            i_ = 0;
        }
    }

private:

    pangolin::Var<double> fps_;

    int displayFrequency_;
    int i_;
    pangolin::basetime lastTime_;

};


} // namespace df
