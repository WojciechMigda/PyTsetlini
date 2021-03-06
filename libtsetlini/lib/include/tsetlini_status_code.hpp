#pragma once

#ifndef LIB_INCLUDE_TSETLINI_STATUS_CODE_HPP_
#define LIB_INCLUDE_TSETLINI_STATUS_CODE_HPP_

#include <utility>
#include <string>

namespace Tsetlini
{

enum StatusCode
{
    S_OK = 0,

    S_BAD_JSON,
    S_VALUE_ERROR,
    S_NOT_FITTED_ERROR,
};

using status_message_t = std::pair<StatusCode, std::string>;


} // namespace Tsetlini


#endif /* LIB_INCLUDE_TSETLINI_STATUS_CODE_HPP_ */
