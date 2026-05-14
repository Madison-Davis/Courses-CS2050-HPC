// test_cuda.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include "Config.h"


// ++++++++ VARIABLES ++++++++ //
const double EPS = 1e-6; // threshold to be considered close
struct BodyGPU;
BodyGPU allocBodyGPU(int N);
// Serial versions
void compute_acceleration_serial(std::vector<Body>&);
void check_collisions_serial(std::vector<Body>&);
void integrate_serial(std::vector<Body>&, double);
// CUDA versions and wrappers
void freeBodyGPU(BodyGPU& d);
void copyBodiesToGPU(const std::vector<Body>& h, BodyGPU& d, int N);
void copyBodiesFromGPU(const BodyGPU& d, int N, std::vector<Body>& bodies);
void compute_acceleration_cuda(BodyGPU d, int N);
void check_collisions_cuda(BodyGPU d, int N);
void integrate_cuda(BodyGPU d, int N, double dt);


// ++++++++ FUNCTIONS ++++++++ //
bool close(double a, double b) {
    double diff = std::abs(a - b);
    double mag  = std::max(std::abs(a), std::abs(b));
    if (mag < 1.0) return diff < 1e-9;              // absolute tol for small values
    return diff / mag < 1e-6;                       // relative tol for large values
}

bool compare_vec3(const Vec3& a, const Vec3& b) {
    return close(a.x, b.x) &&
           close(a.y, b.y) &&
           close(a.z, b.z);
}

bool compare_bodies(const std::vector<Body>& A,
                    const std::vector<Body>& B,
                    const std::string& label)
{
    if (A.size() != B.size()) {
        std::cout << label << ": size mismatch\n";
        return false;
    }

    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i].alive != B[i].alive) {
            std::cout << label << ": alive mismatch at " << i << "\n";
            return false;
        }
        if (!compare_vec3(A[i].pos, B[i].pos)) {
            std::cout << label << ": pos mismatch at " << i
                      << " serial=(" << A[i].pos.x << "," << A[i].pos.y << "," << A[i].pos.z << ")"
                      << " cuda=("   << B[i].pos.x << "," << B[i].pos.y << "," << B[i].pos.z << ")\n";
            return false;
        }
        if (!compare_vec3(A[i].vel, B[i].vel)) {
            std::cout << label << ": vel mismatch at " << i
                      << " serial=(" << A[i].vel.x << "," << A[i].vel.y << "," << A[i].vel.z << ")"
                      << " cuda=("   << B[i].vel.x << "," << B[i].vel.y << "," << B[i].vel.z << ")\n";
            return false;
        }
        if (!compare_vec3(A[i].acc, B[i].acc)) {
            std::cout << label << ": acc mismatch at " << i
                      << " serial=(" << A[i].acc.x << "," << A[i].acc.y << "," << A[i].acc.z << ")"
                      << " cuda=("   << B[i].acc.x << "," << B[i].acc.y << "," << B[i].acc.z << ")"
                      << " diff=("   << std::abs(A[i].acc.x - B[i].acc.x) << ","
                                     << std::abs(A[i].acc.y - B[i].acc.y) << ","
                                     << std::abs(A[i].acc.z - B[i].acc.z) << ")\n";
            return false;
        }
        if (!close(A[i].mass, B[i].mass)) {
            std::cout << label << ": mass mismatch at " << i << "\n";
            return false;
        }
        if (!close(A[i].radius, B[i].radius)) {
            std::cout << label << ": radius mismatch at " << i << "\n";
            return false;
        }
    }
    return true;
}


// ++++++++ GENERATION FUNCTION ++++++++ //
std::vector<Body> make_bodies(int N) {
    std::vector<Body> bodies;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; ++i) {
        Body b;
        b.pos = {dist(rng), dist(rng), dist(rng)};
        b.vel = {dist(rng), dist(rng), dist(rng)};
        b.acc = {0, 0, 0};
        b.mass = 1.0;
        b.radius = 0.01;
        b.alive = true;
        bodies.push_back(b);
    }

    return bodies;
}


// ++++++++ MAIN FUNCTION ++++++++ //
int main() {
    // Variables
    const int N = 100;
    const double dt = 0.01;

    // Create Bodies
    BodyGPU d = allocBodyGPU(N);
    auto base = make_bodies(N);

    // Test 1: Acceleration
    {
        auto serial = base;
        auto cuda   = base;

        compute_acceleration_serial(serial);
        copyBodiesToGPU(cuda, d, N);
        compute_acceleration_cuda(d, N);
        copyBodiesFromGPU(d, N, cuda);

        if (!compare_bodies(serial, cuda, "ACCELERATION")) {
            std::cout << "[ERROR] Acceleration mismatch\n";
            return 1;
        }
        std::cout << "[SUCCESS] Acceleration match\n";
    }

    // Test 2: Collisions
    {
        auto serial = base;
        auto cuda   = base;

        check_collisions_serial(serial);
        copyBodiesToGPU(cuda, d, N);
        check_collisions_cuda(d, N);
        copyBodiesFromGPU(d, N, cuda);

        if (!compare_bodies(serial, cuda, "COLLISION")) {
            std::cout << "[ERROR] Collision mismatch\n";
            return 1;
        }
        std::cout << "[SUCCESS] Collision match\n";
    }

    // Test 3: Integrate
    {
        auto serial = base;
        auto cuda   = base;

        integrate_serial(serial, dt);
        copyBodiesToGPU(cuda, d, N);
        integrate_cuda(d, N, dt);
        copyBodiesFromGPU(d, N, cuda);

        if (!compare_bodies(serial, cuda, "INTEGRATION")) {
            std::cout << "[ERROR] Integration mismatch\n";
            return 1;
        }
        std::cout << "[SUCCESS] Integration match\n";
    }

    freeBodyGPU(d);
    std::cout << "\n[SUCCESS] All CUDA vs Serial tests passed\n";
    return 0;
}