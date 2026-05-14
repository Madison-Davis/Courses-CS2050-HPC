// test_mpi.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include "Config.h"
#include <mpi.h>


// ++++++++ VARIABLES ++++++++ //
const double EPS = 1e-6; // threshold to be considered close
// Serial versions
void compute_acceleration_serial(std::vector<Body>&);
void check_collisions_serial(std::vector<Body>&);
void integrate_serial(std::vector<Body>&, double);
// MPI versions
void compute_acceleration_mpi(std::vector<Body>&, int rank, int size);
void check_collisions_mpi(std::vector<Body>&);
void integrate_mpi(std::vector<Body>&, double, int rank, int size,
                   const std::vector<int>& counts, const std::vector<int>& displs,
                   const std::vector<int>& pos_counts, const std::vector<int>& pos_displs,
                   const std::vector<int>& state_counts, const std::vector<int>& state_displs);


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
int main(int argc, char** argv) {

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Variables
    int N = 100;
    double dt = 0.01;
    
    // Create Bodies On Rank 0
    std::vector<Body> bodies_serial;
    std::vector<Body> bodies_mpi;
    if (rank == 0) {
        bodies_serial = make_bodies(N);
        bodies_mpi = bodies_serial;
    }

    // Broadcast Initial Bodies To Other Ranks
    if (rank != 0) bodies_mpi.resize(N);
    MPI_Bcast(bodies_mpi.data(),    // buffer
            N * sizeof(Body),       // count
            MPI_BYTE,               // datatype
            0,                      // root
            MPI_COMM_WORLD);        // comms

    // Test 1: Acceleration
    if (rank == 0) compute_acceleration_serial(bodies_serial);
    compute_acceleration_mpi(bodies_mpi, rank, size);
    {
    // Compute some helpful vectors before gathering back on rank 0
    // counts: how many bytes rank r contributes
    // displs: byte offset rank r is in the receive buffer
    size_t chunk = N / size;
    std::vector<int> counts(size), displs(size);
    for (int rank = 0; rank < size; ++rank) {
        size_t start = rank * chunk;
        size_t end = (rank == size - 1) ? N : start + chunk;
        counts[rank] = (end - start) * sizeof(Body);
        displs[rank] = start * sizeof(Body);
    }
    // Gather full result, then broadcast updated state to all for next test
    // Allgatherv = Gatherv + Bcast!
    // Allow 'v' for Allgatherv in the case diff ranks deal with diff # bodies
    MPI_Allgatherv(
        MPI_IN_PLACE,           // data is already in spot you wish to recv them in
        0, MPI_DATATYPE_NULL,   // send args ignored for the reason above
        bodies_mpi.data(),      // recv's array: buffer on rank 0 (ignored on other ranks)
        counts.data(),          // recv's array: how many bytes to expect from each rank
        displs.data(),          // recv's array: where in recv buffer to place each rank's data
        MPI_BYTE,               // datatype
        MPI_COMM_WORLD          // comms
    );
    }
    // Compare results on rank 0
    if (rank == 0) {
        if (!compare_bodies(bodies_serial, bodies_mpi, "ACCELERATION")) {
            std::cout << "[ERROR] Acceleration mismatch\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            std::cout << "[SUCCESS] Acceleration match\n";
        }
    }
    
    // Test 2: Collisions
    if (rank == 0) check_collisions_serial(bodies_serial);
    check_collisions_mpi(bodies_mpi);
    // Compare results on rank 0
    if (rank == 0) {
        if (!compare_bodies(bodies_serial, bodies_mpi, "COLLISIONS")) {
            std::cout << "[ERROR] Collision mismatch\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            std::cout << "[SUCCESS] Collisions match\n";
        }
    }
    // Broadcast updated state for next test
    MPI_Bcast(bodies_mpi.data(),    // buffer
            N * sizeof(Body),       // count
            MPI_BYTE,               // datatype
            0,                      // root
            MPI_COMM_WORLD);        // comms

    // Test 3: Integrate
    // Compute some helpful vectors before gathering back on rank 0
    // counts: how many bytes rank r contributes
    // displs: byte offset rank r is in the receive buffer
    size_t chunk = N / size;
    std::vector<int> counts(size), displs(size);
    std::vector<int> pos_counts(size), pos_displs(size);
    std::vector<int> state_counts(size), state_displs(size);
    struct BodyPos { double x, y, z; };                                     // position-only
    struct BodyState { Vec3 vel, acc; double mass, radius; bool alive; };   // essentially everything but position
    for (int r = 0; r < size; ++r) {
        size_t r_start = r * chunk;
        size_t r_end   = (r == size - 1) ? N: r_start + chunk;
        size_t n       = r_end - r_start;
        counts[r]       = n * sizeof(Body);
        displs[r]       = r_start * sizeof(Body);
        pos_counts[r]   = n * sizeof(BodyPos);
        pos_displs[r]   = r_start * sizeof(BodyPos);
        state_counts[r] = n * sizeof(BodyState);
        state_displs[r] = r_start * sizeof(BodyState);
    }
    if (rank == 0) integrate_serial(bodies_serial, dt);
    integrate_mpi(bodies_mpi, dt, rank, size, counts, displs, pos_counts, pos_displs, state_counts, state_displs);
    {
    // Gather full result, then broadcast updated state to all to end with a synchronized test
    // Allgatherv = Gatherv + Bcast!
    // Allow 'v' for Allgatherv in the case diff ranks deal with diff # bodies
    MPI_Allgatherv(
        MPI_IN_PLACE,           // data is already in spot you wish to recv them in
        0, MPI_DATATYPE_NULL,   // send args ignored for the reason above
        bodies_mpi.data(),      // recv's array: buffer on rank 0 (ignored on other ranks)
        counts.data(),          // recv's array: how many bytes to expect from each rank
        displs.data(),          // recv's array: where in recv buffer to place each rank's data
        MPI_BYTE,               // datatype
        MPI_COMM_WORLD          // comms
    );
    }
    // Compare results on rank 0
    if (rank == 0) {
        if (!compare_bodies(bodies_serial, bodies_mpi, "INTEGRATE")) {
            std::cout << "[ERROR] Integration mismatch\n";
        } else {
            std::cout << "[SUCCESS] Integration match\n";
        }
    }

    if (rank == 0) {
        std::cout << "[SUCCESS] All MPI tests passed\n";
    }

    // MPI: finalize
    MPI_Finalize();
    return 0;
}