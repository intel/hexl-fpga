// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __UNROLLER_HPP__
#define __UNROLLER_HPP__

template <size_t it, size_t end>
struct Unroller {
    template <typename Action>
    static void Step(const Action& action) {
        action(std::integral_constant<size_t, it>());
        Unroller<it + 1, end>::Step(action);
    }
};

template <size_t end>
struct Unroller<end, end> {
    template <typename Action>
    static void Step(const Action&) {}
};

#endif /* __UNROLLER_HPP__ */
