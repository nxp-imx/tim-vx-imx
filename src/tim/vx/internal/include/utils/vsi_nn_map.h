/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#ifndef _VSI_NN_MAP_H
#define _VSI_NN_MAP_H

#include "vsi_nn_types.h"
#include "vsi_nn_link_list.h"
#include "vsi_nn_binary_tree.h"

#if defined(__cplusplus)
extern "C"{
#endif

typedef vsi_nn_binary_tree_key_t VSI_PUBLIC_TYPE vsi_nn_map_key_t;

typedef struct _vsi_nn_map_key_list
{
    vsi_nn_link_list_t   link_list;
    vsi_nn_map_key_t     val;
} vsi_nn_map_key_list_t;

typedef struct _vsi_nn_map
{
    int size;
    vsi_nn_map_key_list_t * keys;
    vsi_nn_binary_tree_t  * values;
} VSI_PUBLIC_TYPE vsi_nn_map_t;

OVXLIB_API void vsi_nn_MapInit
    (
    vsi_nn_map_t * map
    );

OVXLIB_API void * vsi_nn_MapGet
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key
    );

OVXLIB_API void vsi_nn_MapAdd
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key,
    void              * value
    );

OVXLIB_API void vsi_nn_MapRemove
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key
    );

OVXLIB_API vsi_bool vsi_nn_MapHasKey
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key
    );

#if defined(__cplusplus)
}
#endif

#endif
