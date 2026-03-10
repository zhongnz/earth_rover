#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const char* host = std::getenv("ORB_SLAM3_SIDECAR_HOST");
    const char* port = std::getenv("ORB_SLAM3_SIDECAR_PORT");

    std::string bind_host = host ? host : "127.0.0.1";
    std::string bind_port = port ? port : "8765";

    std::cerr
        << "orbslam3_sidecar scaffold built, but the real HTTP server and ORB-SLAM3 "
        << "backend wiring are not implemented in this tree yet.\n"
        << "Requested bind target: " << bind_host << ':' << bind_port << "\n"
        << "Next step: replace this stub with a concrete implementation of the "
        << "contract in indoor_nav/slam/protocol.md.\n";

    return 2;
}
