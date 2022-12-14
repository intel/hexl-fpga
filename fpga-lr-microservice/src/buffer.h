// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <queue>
#include <memory>
#include "blobs.h"

struct Buffer {
    enum QType { ConfigQ = 0, InputQ, OutputQ, NumQs };
    std::queue<std::shared_ptr<Blob>> queue[NumQs];
};

class Party {
public:
    Party() {}
    explicit Party(std::shared_ptr<Buffer>& buffer) : buffer_(buffer) {}
    virtual ~Party() {}

    virtual bool setup() = 0;
    virtual void configure() = 0;
    virtual void process() = 0;
    virtual void teardown() = 0;

protected:
    std::shared_ptr<Buffer> buffer_;
};
