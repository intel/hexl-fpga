
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipe_array.hpp"

#define defLongPipe(pipeName, dataType, length)                              \
    using pipeName = sycl::ext::intel::pipe<class NameClass4Pipe_##pipeName, \
                                            dataType, (length)>;
#define defPipe(pipeName, dataType) \
    using pipeName =                \
        sycl::ext::intel::pipe<class NameClass4Pipe_##pipeName, dataType>;
#define defPipe1d(pipeName, dataType, depth, dim1)                         \
    using pipeName = PipeArray<class NameClass2dPipe_##pipeName, dataType, \
                               (depth), (dim1)>;
#define defPipe2d(pipeName, dataType, depth, dim1, dim2)                   \
    using pipeName = PipeArray<class NameClass2dPipe_##pipeName, dataType, \
                               (depth), (dim1), (dim2)>;
#define defPipe3d(pipeName, dataType, depth, dim1, dim2, dim3)             \
    using pipeName = PipeArray<class NameClass3dPipe_##pipeName, dataType, \
                               (depth), (dim1), (dim2), (dim3)>;
#define defPipe4d(pipeName, dataType, depth, dim1, dim2, dim3, dim4)       \
    using pipeName = PipeArray<class NameClass4dPipe_##pipeName, dataType, \
                               (depth), (dim1), (dim2), (dim3), (dim4)>;
