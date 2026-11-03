/* entropic.h — Umbrella header for the Entropic procedural generation library
 *
 * Includes the full Entropic toolkit:
 *   - wfc.h       Wave Function Collapse solver (C11)
 *   - gray_codes.h  Gray code variants (C99)
 *   - sobol.h     Sobol quasi-random sequences (C99)
 *   - wfcpp.hpp   C++17 RAII wrapper (C++ only)
 *
 *   #define WFC_IMPLEMENTATION   // in exactly one .c/.cpp file
 *   #include <entropic/entropic.h>
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025-2026 StraylightRunMOO
 */

#ifndef ENTROPIC_H
#define ENTROPIC_H

#include "wfc.h"
#include "gray_codes.h"
#include "sobol.h"

#ifdef __cplusplus
#include "wfcpp.hpp"
#endif

#endif /* ENTROPIC_H */
