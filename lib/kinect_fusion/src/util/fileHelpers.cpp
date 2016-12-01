#include <df/util/fileHelpers.h>

namespace df {

std::string compileDirectory() {
    static std::string dir = COMPILE_DIR;
    return dir;
}

} // namespace df
