#include <df/util/args.h>

#include <getopt.h>

#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

#include <iostream>

namespace df {

void OptParse::registerOptionGeneric(const std::string flag, void * valuePtr, const unsigned char shorthand, const bool required, const ArgType type) {
    flags_.push_back(flag);
    valuePtrs_.push_back(valuePtr);
    shorthands_.push_back(shorthand);
    requireds_.push_back(required);
    types_.push_back(type);
}

void OptParse::registerOption(const std::string flag, std::string & value, const unsigned char shorthand, const bool required) {
    registerOptionGeneric(flag,&value,shorthand,required,String);
}

void OptParse::registerOption(const std::string flag, int & value, const unsigned char shorthand, const bool required) {
    registerOptionGeneric(flag,&value,shorthand,required,Integral);
}

void OptParse::registerOption(const std::string flag, float & value, const unsigned char shorthand, const bool required) {
    registerOptionGeneric(flag,&value,shorthand,required,FloatingPoint);
}

void OptParse::registerOption(const std::string flag, bool & value, const unsigned char shorthand, const bool required) {
    registerOptionGeneric(flag,&value,shorthand,required,Boolean);
}


int OptParse::parseOptions(int & argc, char * * & argv) {

    const std::size_t nArgs = flags_.size();
    if (nArgs >= std::numeric_limits<unsigned char>::max()) {
        throw std::runtime_error("too many arguments");
    }

    std::vector<struct option> options(nArgs+1);
    std::string shorthandString = "";

    std::set<unsigned char> usedShorthands;

    for (std::size_t i = 0; i < nArgs; ++i) {

        const unsigned char shorthand = shorthands_[i];
        if (shorthands_[i] < std::numeric_limits<unsigned char>::max()) {
            // check to make sure shorthand has not been taken
            if (usedShorthands.find(shorthands_[i]) != usedShorthands.end()) {
                throw std::runtime_error("the shorthand " + std::to_string(shorthand) + " was used more than once");
            }

            shorthandString.append( { shorthand } );
            if (types_[i] != Boolean) {
                shorthandString.append({':'});
            }
        }

        options[i].name = flags_[i].c_str();
        options[i].has_arg = types_[i] == Boolean ? no_argument : required_argument;
        options[i].flag = nullptr;
        options[i].val = shorthand;
    }

    std::memset(&options.back(),0,sizeof(struct option));

    // assign unused values to flags with no shorthand
    int nextShorthand = 0;
    for (std::size_t i = 0; i < nArgs; ++i) {
        if (shorthands_[i] == std::numeric_limits<unsigned char>::max()) {
            while (usedShorthands.find(nextShorthand) != usedShorthands.end()) {
                nextShorthand++;
            }
            shorthands_[i] = nextShorthand;
            options[i].val = shorthands_[i];
            ++nextShorthand;
        }
    }

    int c;

    while (true) {

        int optionIndex = 0;
        c = getopt_long(argc,argv,shorthandString.c_str(),options.data(),&optionIndex);

        if (c == -1) {
            break;
        }

        std::size_t i;
        for (i = 0; i < nArgs; ++i) {
            if (shorthands_[i] == c) {

                switch (types_[i]) {
                case Integral:
                    *reinterpret_cast<int *>(valuePtrs_[i]) = atoi(optarg);
                    break;
                case FloatingPoint:
                    *reinterpret_cast<float *>(valuePtrs_[i]) = atof(optarg);
                    break;
                case Boolean:
                    *reinterpret_cast<bool *>(valuePtrs_[i]) = true;
                    break;
                case String:
                    *reinterpret_cast<std::string *>(valuePtrs_[i]) = std::string(optarg);
                    break;
                }

                requireds_[i] = false;
                break;
            }
        }

        if (i == nArgs) {
            throw std::runtime_error("unrecognized argument");
        }

    }

    for (std::size_t i = 0; i < nArgs; ++i) {
        if (requireds_[i]) {
            throw std::runtime_error("did not find required argument '" + flags_[i] + "'");
        }
    }

    return optind;

}

} // namespace df
