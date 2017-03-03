/*
Copyright (c) 2016, TU Dresden
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <chrono>

/**
 * @brief Class for time measurements.
 */
class StopWatch
{
public:
    /**
     * @brief Construction. Initializes the stop watch.
     */
    StopWatch(){ init(); }
  
    /**
     * @brief Initialization. Starts the time measurement.
     * 
     * @return void
     */
    void init()
    {
	start = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief Stops and restarts the time measurement.
     * 
     * @return float The time in ms since the last init or stop call.
     */
    float stop()
    {
	std::chrono::high_resolution_clock::time_point now;
	now = std::chrono::high_resolution_clock::now();
	
	std::chrono::high_resolution_clock::duration duration = now - start;
	
	start = now;
	
	return static_cast<float>(
	    1000.0 * std::chrono::duration_cast<std::chrono::duration<double>>(
	    duration).count());
    }
    
private:
    std::chrono::high_resolution_clock::time_point start; // start time of the current measurement.
};