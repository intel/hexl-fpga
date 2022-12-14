// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "client.h"
#include "server.h"

class Service {
public:
    explicit Service(Client& client, Server& server)
        : client_(client), server_(server) {}

    bool setup() {
        bool ok = server_.setup();
        ok |= client_.setup();
        return ok;
    }

    void configure() {
        client_.configure();
        server_.configure();
    }

    void process() {
        while (client_.moreInputs()) {
            client_.process();
            server_.process();
        }
    }

    void teardown() {
        // destructors teardown the service
    }

private:
    Client& client_;
    Server& server_;
};
