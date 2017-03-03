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

#include <string>
#include <vector>

/** General utility functions.*/

// makros for coloring console output
#define GREENTEXT(output) "\x1b[32;1m" << output << "\x1b[0m"
#define REDTEXT(output) "\x1b[31;1m" << output << "\x1b[0m" 
#define BLUETEXT(output) "\x1b[34;1m" << output << "\x1b[0m" 
#define YELLOWTEXT(output) "\x1b[33;1m" << output << "\x1b[0m" 

/**
 * @brief Splits a string using the given delimiter character.
 * 
 * @param s String to split.
 * @param delim Delimiter used to split the string into chunks. The delimiter will be removed.
 * @return std::vector< std::string, std::allocator< void > > List of string chunks.
 */
std::vector<std::string> split(const std::string& s, char delim);

/**
 * @brief Splits a string at spaces.
 * 
 * @param s String to split.
 * @return std::vector< std::string, std::allocator< void > > List of string chunks.
 */
std::vector<std::string> split(const std::string& s);

/**
 * @brief Splits a given string in two parts. The split location is before the first number from the right.
 * 
 * @param s String to split.
 * @return std::pair< std::string, std::string > Two parts of the string.
 */
std::pair<std::string, std::string> splitOffDigits(std::string s);

/**
 * @brief Checks whether a string is ending with the key.
 * 
 * @param str String to check.
 * @param key Key to look for a the end of the string.
 * @return bool True if the string ends with the key.
 */
bool endsWith(std::string str, std::string key);

/**
 * @brief Converts a integer number to a string. The string can be filled with leading zeros.
 * 
 * @param number Integer to convert.
 * @param minLength String is padded with leading zeros to achieve this minimal length. Defaults to 0.
 * @return std::string
 */
std::string intToString(int number, int minLength = 0);

/**
* @brief Converts a floating point number to a string.
* 
* @param number Number to convert.
* @return std::string
*/
std::string floatToString(float number);
 
/**
 * @brief Clamps a value at the given min and max values.
 * 
 * @param val Value to clamp.
 * @param min_val Minimal allowed value.
 * @param max_val Maximal allowed value.
 * @return int Clamped value.
 */
int clamp(int val, int min_val, int max_val);

/**
 * @brief Returns a list of directories contained under the given path. The directories are full paths, i.e. they contain the base path.
 * 
 * @param basePath Path were the directories lie.
 * @return std::vector< std::string, std::allocator< void > > List of directories (full paths).
 */
std::vector<std::string> getSubPaths(std::string basePath);

/**
 * @brief Returns a list of files with a given extension contained under the given path. Files are returned including the full path.
 * 
 * @param path Path were the files lie.
 * @param ext Only files with this extension will be returned.
 * @param silent The method will print a message in case the given path does not exist. This can be supressed by setting silent to true. Defaults to false.
 * @return std::vector< std::string, std::allocator< void > > List of files (contain the full path).
 */
std::vector<std::string> getFiles(std::string path, std::string ext, bool silent = false);