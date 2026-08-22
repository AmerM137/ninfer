#include "ninfer/types.h"

#include <utility>

namespace ninfer {

CancellationView::CancellationView(std::function<bool()> requested)
    : callback(std::move(requested)) {}

bool CancellationView::requested() const { return callback && callback(); }

} // namespace ninfer
