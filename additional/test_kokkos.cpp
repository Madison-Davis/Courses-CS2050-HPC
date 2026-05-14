// test_kokkos.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include "Config.h"
#include <Kokkos_Core.hpp>


// ++++++++ VARIABLES ++++++++ //
const double EPS = 1e-6;  // threshold to be considered close
// Serial versions
void compute_acceleration_serial(std::vector<Body>&);
void check_collisions_serial(std::vector<Body>&);
void integrate_serial(std::vector<Body>&, double);
// Kokkos versions
void compute_acceleration_kokkos(int,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<double*>,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<double*>,
    Kokkos::View<double*>, Kokkos::View<int*>);
void check_collisions_kokkos(int,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<double*>,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<double*>,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<int*>);
void integrate_kokkos(int, double,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<double*>,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<double*>,
    Kokkos::View<double*>, Kokkos::View<double*>, Kokkos::View<double*>,
    Kokkos::View<double*>, Kokkos::View<int*>);


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
        b.pos    = {dist(rng), dist(rng), dist(rng)};
        b.vel    = {dist(rng), dist(rng), dist(rng)};
        b.acc    = {0, 0, 0};
        b.mass   = 1.0;
        b.radius = 0.01;
        b.alive  = true;
        bodies.push_back(b);
    }

    return bodies;
}


// ++++++++ HELPER FUNCTIONS ++++++++ //
struct KokkosViews {
    // Struct that bundles all device Views together
    Kokkos::View<double*> pos_x, pos_y, pos_z;
    Kokkos::View<double*> vel_x, vel_y, vel_z;
    Kokkos::View<double*> acc_x, acc_y, acc_z;
    Kokkos::View<double*> mass;
    Kokkos::View<double*> radius;
    Kokkos::View<int*>    alive;
    KokkosViews(int N)
        : pos_x("x",N), pos_y("y",N), pos_z("z",N),
          vel_x("vx",N), vel_y("vy",N), vel_z("vz",N),
          acc_x("ax",N), acc_y("ay",N), acc_z("az",N),
          mass("m",N), radius("r",N), alive("alive",N) {}
};

void load_views(const std::vector<Body>& bodies, KokkosViews& v) {
    // Converts host vector<Body> to device Views
    // Why: serial code stores each body as an array of structs
    // But device works better with structs of arrays

    // Create a mirror View to use for transfer
    int N = bodies.size();
    auto h_x  = Kokkos::create_mirror_view(v.pos_x);
    auto h_y  = Kokkos::create_mirror_view(v.pos_y);
    auto h_z  = Kokkos::create_mirror_view(v.pos_z);
    auto h_vx = Kokkos::create_mirror_view(v.vel_x);
    auto h_vy = Kokkos::create_mirror_view(v.vel_y);
    auto h_vz = Kokkos::create_mirror_view(v.vel_z);
    auto h_ax = Kokkos::create_mirror_view(v.acc_x);
    auto h_ay = Kokkos::create_mirror_view(v.acc_y);
    auto h_az = Kokkos::create_mirror_view(v.acc_z);
    auto h_m  = Kokkos::create_mirror_view(v.mass);
    auto h_r  = Kokkos::create_mirror_view(v.radius);
    auto h_a  = Kokkos::create_mirror_view(v.alive);

    // Fill in what the host will transfer over
    for (int i = 0; i < N; ++i) {
        h_x(i)  = bodies[i].pos.x;  h_y(i)  = bodies[i].pos.y;  h_z(i)  = bodies[i].pos.z;
        h_vx(i) = bodies[i].vel.x;  h_vy(i) = bodies[i].vel.y;  h_vz(i) = bodies[i].vel.z;
        h_ax(i) = bodies[i].acc.x;  h_ay(i) = bodies[i].acc.y;  h_az(i) = bodies[i].acc.z;
        h_m(i)  = bodies[i].mass;
        h_r(i)  = bodies[i].radius;
        h_a(i)  = bodies[i].alive ? 1 : 0;
    }

    // Copy from host to device
    Kokkos::deep_copy(v.pos_x, h_x);  Kokkos::deep_copy(v.pos_y, h_y);  Kokkos::deep_copy(v.pos_z, h_z);
    Kokkos::deep_copy(v.vel_x, h_vx); Kokkos::deep_copy(v.vel_y, h_vy); Kokkos::deep_copy(v.vel_z, h_vz);
    Kokkos::deep_copy(v.acc_x, h_ax); Kokkos::deep_copy(v.acc_y, h_ay); Kokkos::deep_copy(v.acc_z, h_az);
    Kokkos::deep_copy(v.mass,  h_m);  Kokkos::deep_copy(v.radius, h_r); Kokkos::deep_copy(v.alive, h_a);
}

std::vector<Body> read_views(KokkosViews& v, int N) {
    // Converts device Views to host vector<Body>
    // Why: serial code stores each body as an array of structs
    // But device works better with structs of arrays

    // Create a mirror View to use as recipient of transfer
    auto h_x  = Kokkos::create_mirror_view(v.pos_x);
    auto h_y  = Kokkos::create_mirror_view(v.pos_y);
    auto h_z  = Kokkos::create_mirror_view(v.pos_z);
    auto h_vx = Kokkos::create_mirror_view(v.vel_x);
    auto h_vy = Kokkos::create_mirror_view(v.vel_y);
    auto h_vz = Kokkos::create_mirror_view(v.vel_z);
    auto h_ax = Kokkos::create_mirror_view(v.acc_x);
    auto h_ay = Kokkos::create_mirror_view(v.acc_y);
    auto h_az = Kokkos::create_mirror_view(v.acc_z);
    auto h_m  = Kokkos::create_mirror_view(v.mass);
    auto h_r  = Kokkos::create_mirror_view(v.radius);
    auto h_a  = Kokkos::create_mirror_view(v.alive);

    // Perform a deep copy from device to host
    Kokkos::deep_copy(h_x,  v.pos_x); Kokkos::deep_copy(h_y,  v.pos_y); Kokkos::deep_copy(h_z,  v.pos_z);
    Kokkos::deep_copy(h_vx, v.vel_x); Kokkos::deep_copy(h_vy, v.vel_y); Kokkos::deep_copy(h_vz, v.vel_z);
    Kokkos::deep_copy(h_ax, v.acc_x); Kokkos::deep_copy(h_ay, v.acc_y); Kokkos::deep_copy(h_az, v.acc_z);
    Kokkos::deep_copy(h_m,  v.mass);  Kokkos::deep_copy(h_r,  v.radius); Kokkos::deep_copy(h_a,  v.alive);

    // Fill in what the host will receive
    std::vector<Body> bodies(N);
    for (int i = 0; i < N; ++i) {
        bodies[i].pos    = {h_x(i),  h_y(i),  h_z(i)};
        bodies[i].vel    = {h_vx(i), h_vy(i), h_vz(i)};
        bodies[i].acc    = {h_ax(i), h_ay(i), h_az(i)};
        bodies[i].mass   = h_m(i);
        bodies[i].radius = h_r(i);
        bodies[i].alive  = (h_a(i) != 0);
    }
    return bodies;
}

// ++++++++ MAIN FUNCTION ++++++++ //
int main(int argc, char* argv[]) {

    // Kokkos: initialize
    Kokkos::initialize(argc, argv);
    {
        // Variables
        const int N      = 100;
        const double dt  = 0.01;

        // Create Bodies
        auto bodies_serial = make_bodies(N);
        auto bodies_kokkos = bodies_serial;
        KokkosViews v(N);
        load_views(bodies_kokkos, v);

        // Test 1: Acceleration
        compute_acceleration_serial(bodies_serial);
        compute_acceleration_kokkos(N,
            v.pos_x, v.pos_y, v.pos_z,
            v.acc_x, v.acc_y, v.acc_z,
            v.mass, v.alive);
        Kokkos::fence();
        bodies_kokkos = read_views(v, N);

        if (!compare_bodies(bodies_serial, bodies_kokkos, "ACCELERATION")) {
            std::cout << "[ERROR] Acceleration mismatch\n";
            Kokkos::finalize(); return 1;
        } else {
            std::cout << "[SUCCESS] Acceleration match\n";
        }

        // Test 2: Collisions
        bodies_serial = make_bodies(N);
        bodies_kokkos = bodies_serial;
        load_views(bodies_kokkos, v);

        check_collisions_serial(bodies_serial);
        check_collisions_kokkos(N, v.pos_x, v.pos_y, v.pos_z, v.vel_x, v.vel_y, v.vel_z, v.mass, v.radius, v.alive);
        Kokkos::fence();
        bodies_kokkos = read_views(v, N);

        if (!compare_bodies(bodies_serial, bodies_kokkos, "COLLISIONS")) {
            std::cout << "[ERROR] Collision mismatch\n";
            Kokkos::finalize(); return 1;
        } else {
            std::cout << "[SUCCESS] Collisions match\n";
        }

        // Test 3: Integrate
        bodies_serial = make_bodies(N);
        bodies_kokkos = bodies_serial;
        load_views(bodies_kokkos, v);
        
        integrate_serial(bodies_serial, dt);
        integrate_kokkos(N, dt,
            v.pos_x, v.pos_y, v.pos_z,
            v.vel_x, v.vel_y, v.vel_z,
            v.acc_x, v.acc_y, v.acc_z,
            v.mass, v.alive);
        Kokkos::fence();
        bodies_kokkos = read_views(v, N);

        if (!compare_bodies(bodies_serial, bodies_kokkos, "INTEGRATE")) {
            std::cout << "[ERROR] Integration mismatch\n";
            Kokkos::finalize(); return 1;
        } else {
            std::cout << "[SUCCESS] Integration match\n";
        }

        std::cout << "\n[SUCCESS] All tests passed\n";
    }
    Kokkos::finalize();
    return 0;
}