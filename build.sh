#!/usr/bin/env bash

project_dir=${PWD}
install=${project_dir}/install
build_type=Release

rm -rf ${install} tools

. build_server.sh
. build_client.sh
. build_service.sh
