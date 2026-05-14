// test_openmp.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include "Config.h"
#include <omp.h>


// ++++++++ VARIABLES ++++++++ //
const double EPS = 1e-6; // threshold to be considered close
// Serial versions
void compute_acceleration_serial(std::vector<Body>&);
void check_collisions_serial(std::vector<Body>&);
void integrate_serial(std::vector<Body>&, double);
// OpenMP versions
void compute_acceleration_openmp(std::vector<Body>&);
void check_collisions_openmp(std::vector<Body>&);
void integrate_openmp(std::vector<Body>&, double);


// ++++++++ FUNCTIONS ++++++++ //
bool close(double a, double b) {
    return std::abs(a - b) < EPS;
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
            std::cout << label << ": pos mismatch at " << i << "\n";
            return false;
        }

        if (!compare_vec3(A[i].vel, B[i].vel)) {
            std::cout << label << ": vel mismatch at " << i << "\n";
            return false;
        }

        if (!compare_vec3(A[i].acc, B[i].acc)) {
            std::cout << label << ": acc mismatch at " << i << "\n";
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
        b.acc = {0,0,0};
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
    const int N = 100;      // keep small for debugging
    const double dt = 0.01;

    // Create Bodies
    auto bodies_serial = make_bodies(N);
    auto bodies_openmp = bodies_serial;

    // Test 1: Acceleration
    compute_acceleration_serial(bodies_serial);
    compute_acceleration_openmp(bodies_openmp);
    if (!compare_bodies(bodies_serial, bodies_openmp, "ACCELERATION")) {
        std::cout << "[ERORR] Acceleration mismatch\n";
        return 1;
    } else {
        std::cout << "[SUCCESS] Acceleration match\n";
    }

    // Test 2: Collisions
    check_collisions_serial(bodies_serial);
    check_collisions_openmp(bodies_openmp);
    if (!compare_bodies(bodies_serial, bodies_openmp, "COLLISIONS")) {
        std::cout << "[ERORR] Collision mismatch\n";
        return 1;
    } else {
        std::cout << "[SUCCESS] Collisions match\n";
    }

    // Test 3: Integrate
    integrate_serial(bodies_serial, dt);
    integrate_openmp(bodies_openmp, dt);
    if (!compare_bodies(bodies_serial, bodies_openmp, "INTEGRATE")) {
        std::cout << "[ERROR] Integration mismatch\n";
        return 1;
    } else {
        std::cout << "[SUCCESS] Integration match\n";
    }

    std::cout << "[SUCCESS] All tests passed\n";
    return 0;
}