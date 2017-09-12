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

#include <random>

/** Classes and methods for generating random numbers in multi-threaded programs. */

/**
 * @brief Provides random numbers for multiple threads.
 * 
 * Singelton class. Holds a random number generator for each thread and gives random numbers for the current thread.
 */
class ThreadRand
{
public:
  /**
   * @brief Returns a random integer (uniform distribution).
   * 
   * @param min Minimum value of the random integer (inclusive).
   * @param max Maximum value of the random integer (exclusive).
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return int Random integer value.
   */
  static int irand(int min, int max, int tid = -1);
  
  /**
   * @brief Returns a random double value (uniform distribution).
   * 
   * @param min Minimum value of the random double (inclusive).
   * @param max Maximum value of the random double (inclusive).
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random double value.
   */
  static double drand(double min, double max, int tid = -1);
  
  /**
   * @brief Returns a random double value (Gauss distribution).
   * 
   * @param mean Mean of the Gauss distribution to sample from.
   * @param stdDev Standard deviation of the Gauss distribution to sample from.
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random double value.
   */
  static double dgauss(double mean, double stdDev, int tid = -1);
    
  /**
   * @brief Re-Initialize the object with the given seed.
   * 
   * @param seed Seed to initialize the random number generators (seed is incremented by one for each generator).
   * @return void
   */
  static void forceInit(unsigned seed);
  
private:  
  /**
   * @brief List of random number generators. One for each thread.
   * 
   */
  static std::vector<std::mt19937> generators;
  /**
   * @brief True if the class has been initialized already
   */
  static bool initialised;
  /**
   * @brief Initialize class with the given seed.
   * 
   * Method will create a random number generator for each thread. The given seed 
   * will be incremented by one for each generator. This methods is automatically 
   * called when this calss is used the first time.
   * 
   * @param seed Optional parameter. Seed to be used when initializing the generators. Will be incremented by one for each generator.
   * @return void
   */
  static void init(unsigned seed = 1305);
};

/**
  * @brief Returns a random integer (uniform distribution).
  * 
  * This method used the ThreadRand class.
  * 
  * @param min Minimum value of the random integer (inclusive).
  * @param max Maximum value of the random integer (exclusive).
  * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
  * @return int Random integer value.
  */
int irand(int incMin, int excMax, int tid = -1);
/**
  * @brief Returns a random double value (uniform distribution).
  * 
  * This method used the ThreadRand class.
  * 
  * @param min Minimum value of the random double (inclusive).
  * @param max Maximum value of the random double (inclusive).
  * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
  * @return double Random double value.
  */
double drand(double incMin, double incMax, int tid = -1);

  /**
   * @brief Returns a random integer value (Gauss distribution).
   * 
   * This method used the ThreadRand class.
   * 
   * @param mean Mean of the Gauss distribution to sample from.
   * @param stdDev Standard deviation of the Gauss distribution to sample from.
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random integer value.
   */
int igauss(int mean, int stdDev, int tid = -1);

  /**
   * @brief Returns a random double value (Gauss distribution).
   * 
   * This method used the ThreadRand class.
   * 
   * @param mean Mean of the Gauss distribution to sample from.
   * @param stdDev Standard deviation of the Gauss distribution to sample from.
   * @param tid Optional parameter. ID of the thread to use. If not given, the method will obtain the thread ID itself.
   * @return double Random double value.
   */
double dgauss(double mean, double stdDev, int tid = -1);
