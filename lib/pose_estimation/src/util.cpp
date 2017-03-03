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

#include "util.h"

#include <iterator>
#include <sstream>
#include <iostream>

#include <algorithm>
#include <dirent.h>

std::vector<std::string> split(const std::string& s, char delim) 
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    
    while (std::getline(ss, item, delim)) elems.push_back(item);
    
    return elems;
}

std::vector<std::string> split(const std::string& s) 
{
    std::istringstream iss(s);
    std::vector<std::string> elems;

    std::copy(
	std::istream_iterator<std::string>(iss),
	std::istream_iterator<std::string>(),
	std::back_inserter<std::vector<std::string>>(elems));

    return elems;
}

std::pair<std::string, std::string> splitOffDigits(std::string s)
{
    // find first number
    int splitIndex = -1;
    for(int i = 0; i < (int)s.length(); i++)
    {
	char c = s[i];
	if(('0' <= c && c <= '9') || (c == '.'))
	{
	    splitIndex = i;
	    break;
	}
    }
    
    // split before first number
    if(splitIndex == -1)
	return std::pair<std::string,std::string>(s, "");
    else
	return std::pair<std::string,std::string>(s.substr(0,splitIndex), s.substr(splitIndex, s.length() - splitIndex));
}

bool endsWith(std::string str, std::string key)
{
    size_t keylen = key.length();
    size_t strlen = str.length();

    if(keylen <= strlen)
	return str.substr(strlen - keylen, keylen) == key;
    else 
	return false;
}

std::string intToString(int number, int minLength)
{
   std::stringstream ss; //create a stringstream
   ss << number; //add number to the stream
   std::string out = ss.str();
   while((int)out.length() < minLength) out = "0" + out;
   return out; //return a string with the contents of the stream
}

std::string floatToString(float number)
{
   std::stringstream ss; //create a stringstream
   ss << number; //add number to the stream
   return ss.str(); //return a string with the contents of the stream
}

int clamp(int val, int min_val, int max_val)
{
    return std::max(min_val, std::min(max_val, val));
}

std::vector<std::string> getSubPaths(std::string path)
{
    std::vector<std::string> subPaths;  
  
    DIR *dir = opendir(path.c_str());
    struct dirent *ent;
    
    if(dir != NULL) 
    {
	while((ent = readdir(dir)) != NULL) 
	{
	    std::string entry = ent->d_name;
	    if(entry.find(".") == std::string::npos)
		subPaths.push_back(path + entry);
	}
	closedir(dir);
    } 
    else 
	std::cout << REDTEXT("Could not open directory: ") << path << std::endl;

    std::sort(subPaths.begin(), subPaths.end());
    return subPaths;
}

std::vector<std::string> getFiles(std::string path, std::string ext, bool silent)
{
    std::vector<std::string> files;  
  
    DIR *dir = opendir(path.c_str());
    struct dirent *ent;
    
    if(dir != NULL) 
    {
	while((ent = readdir(dir)) != NULL) 
	{
	    std::string entry = ent->d_name;
	    if(endsWith(entry, ext))
		files.push_back(path + entry);
	}
	closedir(dir);
    } 
    else 
	if(!silent) std::cout << REDTEXT("Could not open directory: ") << path << std::endl;

    std::sort(files.begin(), files.end());
    return files;  
}