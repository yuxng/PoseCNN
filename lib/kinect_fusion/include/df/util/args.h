#pragma once

#include <string>
#include <vector>

namespace df {

class OptParse {
public:

    void registerOption(const std::string flag, std::string & value, const unsigned char shorthand = -1, const bool required = false);

    void registerOption(const std::string flag, int & value, const unsigned char shorthand = -1, const bool required = false);

    void registerOption(const std::string flag, float & value, const unsigned char shorthand = -1, const bool required = false);

    void registerOption(const std::string flag, bool & value, const unsigned char shorthand = -1, const bool required = false);

    int parseOptions(int & argc, char * * & argv);

private:

    enum ArgType {
        Integral,
        FloatingPoint,
        Boolean,
        String
    };

    std::vector<std::string> flags_;
    std::vector<unsigned char> shorthands_;
    std::vector<bool> requireds_;
    std::vector<ArgType> types_;
    std::vector<void *> valuePtrs_;

    void registerOptionGeneric(const std::string flag, void * valuePtr, const unsigned char shorthand, const bool required, const ArgType type);
};

} // namespace df
