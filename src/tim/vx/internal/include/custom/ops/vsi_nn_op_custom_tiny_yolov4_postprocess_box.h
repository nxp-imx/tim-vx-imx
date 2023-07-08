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

#ifndef _VSI_NN_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS_BOX_H
#define _VSI_NN_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS_BOX_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _vsi_nn_custom_tiny_yolov4_postprocess_box_param
{
    struct _custom_tiny_yolov4_postprocess_box_local_data_t* local;
    // Add parameters here
    float bias_0;
    float bias_1;
} vsi_nn_custom_tiny_yolov4_postprocess_box_param;
_compiler_assert(offsetof(vsi_nn_custom_tiny_yolov4_postprocess_box_param, local) == 0, \
    vsi_nn_custom_tiny_yolov4_postprocess_box_h );

#ifdef __cplusplus
}
#endif

#endif

