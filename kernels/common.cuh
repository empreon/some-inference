#pragma once

#ifndef COMMON_CUH
#define COMMON_CUH

#define ROW_MAJOR_INDEX(row, col, width) ((row) * (width) + (col))
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

#endif  // COMMON_CUH
