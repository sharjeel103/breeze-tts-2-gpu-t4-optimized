#include "breeze/model.h"

namespace breeze {

bool BreezeModel::load(const std::string & path, bool prefer_gpu, int device_id) {
    backend.init(prefer_gpu, device_id);
    if (!gg.load(path, backend)) return false;
    cfg = parse_config(gg);
    if (!tok.load(gg)) return false;
    return true;
}

void BreezeModel::free() {
    gg.free();
    backend.free();
}

}
