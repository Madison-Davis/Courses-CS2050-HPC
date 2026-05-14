// run_cuda.cpp


// ++++++++ IMPORTS ++++++++ //
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "Config.h"
#include <cuda_runtime.h>


// ++++++++ VARIABLES ++++++++ //
#ifndef TEST_MODE
std::string output_positions = "cuda_positions.csv";
std::string output_timing = "cuda_timing.csv";
#endif
const int THREADS = 256;


// ++++++++ HELPER KERNELS ++++++++ //
__global__ void compute_acceleration_kernel(BodyGPU b, int N, double G, double planet_mass) {
    // Compute acceleration due to planet + other bodies
    int i = blockIdx.x * blockDim.x + threadIdx.x; // compute which body this thread handles
    if (i >= N || !b.alive[i]) return;  // guard extra threads
    double xi = b.pos_x[i];
    double yi = b.pos_y[i];
    double zi = b.pos_z[i];
    double axi = 0, ayi = 0, azi = 0;

    // Compute acceleration of central planet at origin
    double rx = -xi, ry = -yi, rz = -zi;
    double dist = sqrt(rx*rx + ry*ry + rz*rz);
    if (dist > 0) {
        double inv = G * planet_mass / (dist * dist * dist);
        axi += rx * inv;
        ayi += ry * inv;
        azi += rz * inv;
    }

    // Compute acceleration of other bodies
    for (int j = 0; j < N; ++j) {
        if (j == i || !b.alive[j]) continue;
        double dx = b.pos_x[j] - xi;
        double dy = b.pos_y[j] - yi;
        double dz = b.pos_z[j] - zi;
        double d = sqrt(dx*dx + dy*dy + dz*dz);
        if (d > 0) {
            double inv = G * b.mass[j] / (d * d * d);
            axi += dx * inv;
            ayi += dy * inv;
            azi += dz * inv;
        }
    }

    b.acc_x[i] = axi;
    b.acc_y[i] = ayi;
    b.acc_z[i] = azi;
}

__global__ void check_collisions_kernel(BodyGPU b, int N, double planet_r) {
    // Collision detection between planet + other bodies
    int i = blockIdx.x * blockDim.x + threadIdx.x; // compute which body this thread handles
    if (i >= N || !b.alive[i]) return;  // guard extra threads (I launch more threads than bodies)
    double x = b.pos_x[i], y = b.pos_y[i], z = b.pos_z[i];
    double d = sqrt(x*x + y*y + z*z);
    if (d <= planet_r) {
        b.alive[i] = 0;
    }
}

__global__ void integrate_positions_kernel(BodyGPU b, int N, double dt) {
    // Velocity Verlet integration: compute new positions
    int i = blockIdx.x * blockDim.x + threadIdx.x; // compute which body this thread handles
    if (i >= N || !b.alive[i]) return;  // guard extra threads (I launch more threads than bodies)
    b.pos_x[i] += b.vel_x[i] * dt + 0.5 * b.acc_x[i] * dt * dt;
    b.pos_y[i] += b.vel_y[i] * dt + 0.5 * b.acc_y[i] * dt * dt;
    b.pos_z[i] += b.vel_z[i] * dt + 0.5 * b.acc_z[i] * dt * dt;
}

__global__ void integrate_velocity_kernel(BodyGPU b, int N, double dt,
                                               const double* old_ax,
                                               const double* old_ay,
                                               const double* old_az) {
    // Velocity Verlet integration: compute new velocities
    int i = blockIdx.x * blockDim.x + threadIdx.x; // compute which body this thread handles
    if (i >= N || !b.alive[i]) return; // guard extra threads (I launch more threads than bodies)
    b.vel_x[i] += (old_ax[i] + b.acc_x[i]) * 0.5 * dt;
    b.vel_y[i] += (old_ay[i] + b.acc_y[i]) * 0.5 * dt;
    b.vel_z[i] += (old_az[i] + b.acc_z[i]) * 0.5 * dt;
}

__global__ void save_acceleration_kernel(BodyGPU b, int N,
                                          double* old_ax, double* old_ay, double* old_az) {
    // Velocity Verlet integration: save original accelerations                                  
    int i = blockIdx.x * blockDim.x + threadIdx.x; // compute which body this thread handles
    if (i >= N || !b.alive[i]) return; // guard extra threads (I launch more threads than bodies)
    old_ax[i] = b.acc_x[i];
    old_ay[i] = b.acc_y[i];
    old_az[i] = b.acc_z[i];
}


// ++++++++ FUNCTIONS ++++++++ //
void compute_acceleration_cuda(BodyGPU d, int N) {
    // Compute acceleration due to planet + other bodies

    // Cuda: determine number of blocks
    // # blocks = # bodies/# threads (+ THREADS - 1 forces round-up, akin to ceil)
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    compute_acceleration_kernel<<<BLOCKS, THREADS>>>(d, N, Config::G, Config::planet_mass);
    cudaDeviceSynchronize();
}

void check_collisions_cuda(BodyGPU d, int N) {
    // Collision detection between planet + other bodies

    // Cuda: determine number of blocks
    // # blocks = # bodies/# threads (+ THREADS - 1 forces round-up, akin to ceil)
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    check_collisions_kernel<<<BLOCKS, THREADS>>>(d, N, Config::planet_radius);
    cudaDeviceSynchronize();
}

void integrate_cuda(BodyGPU d, int N, double dt) {
    // Velocity Verlet integration

    // Cuda: determine number of blocks
    // # blocks = # bodies/# threads (+ THREADS - 1 forces round-up, akin to ceil)
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    // Update position
    integrate_positions_kernel<<<BLOCKS, THREADS>>>(d, N, dt);
    cudaDeviceSynchronize();
    // Save old acceleration before overwriting
    double *old_ax, *old_ay, *old_az;
    cudaMalloc(&old_ax, N * sizeof(double));
    cudaMalloc(&old_ay, N * sizeof(double));
    cudaMalloc(&old_az, N * sizeof(double));
    save_acceleration_kernel<<<BLOCKS, THREADS>>>(d, N, old_ax, old_ay, old_az);
    cudaDeviceSynchronize();
    // Compute new acceleration and update velocity
    compute_acceleration_cuda(d, N);
    integrate_velocity_kernel<<<BLOCKS, THREADS>>>(d, N, dt, old_ax, old_ay, old_az);
    cudaDeviceSynchronize();

    // Cuda: free old accelerations for cleanup
    cudaFree(old_ax);
    cudaFree(old_ay);
    cudaFree(old_az);
}


// ++++++++ MEMORY FUNCTIONS ++++++++ //
BodyGPU allocBodyGPU(int N) {
    // Allocate a BodyGPU struct
    // All pointer fields point to device memory
    BodyGPU d;
    void** ptrs[] = {
        (void**)&d.pos_x, (void**)&d.pos_y, (void**)&d.pos_z,
        (void**)&d.vel_x, (void**)&d.vel_y, (void**)&d.vel_z,
        (void**)&d.acc_x, (void**)&d.acc_y, (void**)&d.acc_z,
        (void**)&d.mass,  (void**)&d.radius
    };
    for (auto p : ptrs) cudaMalloc(p, N * sizeof(double));
    cudaMalloc((void**)&d.alive, N * sizeof(int));
    return d;
}

void copyBodiesToGPU(const std::vector<Body>& bodies, BodyGPU& d, int N) {
    // Copy all relevant data from CPU to GPU
    // Step 1: create host vectors for outgoing data
    std::vector<double> host_pos_x(N), host_pos_y(N), host_pos_z(N), 
                        host_vel_x(N), host_vel_y(N), host_vel_z(N),
                        host_acc_x(N), host_acc_y(N), host_acc_z(N),
                        host_mass(N), host_radius(N);
    std::vector<int>    host_alive(N);

    // Step 2: convert from array of structs to structs of arrays
    for (int i = 0; i < N; ++i) {
        host_pos_x[i] = bodies[i].pos.x;
        host_pos_y[i] = bodies[i].pos.y;
        host_pos_z[i] = bodies[i].pos.z;
        host_vel_x[i] = bodies[i].vel.x;
        host_vel_y[i] = bodies[i].vel.y;
        host_vel_z[i] = bodies[i].vel.z;
        host_acc_x[i] = bodies[i].acc.x;
        host_acc_y[i] = bodies[i].acc.y;
        host_acc_z[i] = bodies[i].acc.z;
        host_mass[i] = bodies[i].mass;
        host_radius[i] = bodies[i].radius;
        host_alive[i] = bodies[i].alive ? 1 : 0;
    }

    // Step 3: use cudaMemcpy to copy from host to device (both doubes and ints)
    std::pair<double*, double*> dfields[] = {
        {d.pos_x, host_pos_x.data()}, {d.pos_y, host_pos_y.data()}, {d.pos_z, host_pos_z.data()},
        {d.vel_x, host_vel_x.data()}, {d.vel_y, host_vel_y.data()}, {d.vel_z, host_vel_z.data()},
        {d.acc_x, host_acc_x.data()}, {d.acc_y, host_acc_y.data()}, {d.acc_z, host_acc_z.data()},
        {d.mass, host_mass.data()},  {d.radius, host_radius.data()}
    };
    for (auto [dev, host] : dfields)
        cudaMemcpy(dev, host, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d.alive, host_alive.data(), N*sizeof(int), cudaMemcpyHostToDevice);
}

void copyBodiesFromGPU(const BodyGPU& d, int N, std::vector<Body>& bodies) {
    // Copy all relevant data from GPU to CPU
    // Step 1: create host vectors for incoming data
    std::vector<double> host_pos_x(N), host_pos_y(N), host_pos_z(N), 
                        host_vel_x(N), host_vel_y(N), host_vel_z(N),
                        host_acc_x(N), host_acc_y(N), host_acc_z(N),
                        host_mass(N), host_radius(N);
    std::vector<int>    host_alive(N);

    // Step 2: use cudaMemcpy to copy from device to host (both doubes and ints)
    std::pair<double*, double*> dfields[] = {
        {d.pos_x, host_pos_x.data()}, {d.pos_y, host_pos_y.data()}, {d.pos_z, host_pos_z.data()},
        {d.vel_x, host_vel_x.data()}, {d.vel_y, host_vel_y.data()}, {d.vel_z, host_vel_z.data()},
        {d.acc_x, host_acc_x.data()}, {d.acc_y, host_acc_y.data()}, {d.acc_z, host_acc_z.data()},
        {d.mass, host_mass.data()},  {d.radius, host_radius.data()}
    };
    for (auto [dev, host] : dfields)
        cudaMemcpy(host, dev, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_alive.data(), d.alive, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Step 3: convert from structs of arrays to arrays of structs
    for (int i = 0; i < N; ++i) {
        bodies[i].pos    = {host_pos_x[i], host_pos_y[i], host_pos_z[i]};
        bodies[i].vel    = {host_vel_x[i], host_vel_y[i], host_vel_z[i]};
        bodies[i].acc    = {host_acc_x[i], host_acc_y[i], host_acc_z[i]};
        bodies[i].mass   = host_mass[i];
        bodies[i].radius = host_radius[i];
        bodies[i].alive  = host_alive[i];
    }
}

void copyPositionsFromGPU(const BodyGPU& d, int N,
                                  std::vector<double>& hx,
                                  std::vector<double>& hy,
                                  std::vector<double>& hz,
                                  std::vector<int>&    ha) {
    // Copy only the positions + alive flags from GPU to CPU
    cudaMemcpy(hx.data(), d.pos_x, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), d.pos_y, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hz.data(), d.pos_z, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), d.alive, N*sizeof(int),    cudaMemcpyDeviceToHost);
}

void freeBodyGPU(BodyGPU& d) {
    // Deallocate body previously allocated on a GPU device
    void* ptrs[] = {
        d.pos_x, d.pos_y, d.pos_z,
        d.vel_x, d.vel_y, d.vel_z,
        d.acc_x, d.acc_y, d.acc_z,
        d.mass,  d.radius, d.alive
    };
    for (auto p : ptrs) cudaFree(p);
}


// ++++++++ MAIN FUNCTION ++++++++ //
#ifndef TEST_MODE
int main() {

    // Step 0: Construct a random number generator
    std::mt19937 rng(42); // Fix seed for reproducibility
    std::uniform_real_distribution<double> dist_radius(Config::MIN_ORBIT_RADIUS, Config::MAX_ORBIT_RADIUS);
    std::uniform_real_distribution<double> dist_angle(0.0, 2*M_PI);
    std::uniform_real_distribution<double> dist_inclination(-0.1, 0.1); // small inclination

    // Step 1: Create all bodies on CPU
    std::vector<Body> bodies;
    int N = Config::NUM_BODIES;
    bodies.reserve(N);
    for (int i = 0; i < N; ++i) {
        double r     = dist_radius(rng);
        double theta = dist_angle(rng);         // Angle in xy-plane
        double phi   = dist_inclination(rng);   // Small inclination
 
        // Set position in 3D spherical coordinates
        Body b;
        b.pos.x = r * std::cos(theta) * std::cos(phi);
        b.pos.y = r * std::sin(theta) * std::cos(phi);
        b.pos.z = r * std::sin(phi);
 
        // Set circular orbit velocity magnitude
        double v_circ = std::sqrt(Config::G * Config::planet_mass / r);

        // Set velocity perpendicular to position vector (approximate)
        // Use simple rotation in xy-plane and z-component for small inclination
        b.vel.x = -v_circ * std::sin(theta);
        b.vel.y =  v_circ * std::cos(theta);
        b.vel.z =  0.0;
        b.acc    = {0, 0, 0};
        b.mass   = Config::BODY_MASS;
        b.radius = Config::BODY_RADIUS;
        b.alive  = true;

        // Add body to the vector of bodies
        bodies.push_back(b);
    }

    // Step 2: Allocate bodies to GPU memory
    BodyGPU d = allocBodyGPU(N);
    copyBodiesToGPU(bodies, d, N);

    // Step 3: Compute initial acceleration of bodies on GPU
    compute_acceleration_cuda(d, N);
    cudaDeviceSynchronize();

    // Step 4: Define Output Files
        // Define output file for positions
        // Define output file for timing
    std::ofstream outfile(output_positions);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file!\n";
        return 1;
    }
    outfile << "time,body,x,y,z\n";

    std::ofstream timefile(output_timing);
    if (!timefile.is_open()) {
        std::cerr << "Error opening timing file!\n";
        return 1;
    }
    timefile << "step,time_sec\n";
    // Cuda: host-side buffers for output
    std::vector<double> host_pos_x(N), host_pos_y(N), host_pos_z(N);
    std::vector<int>    host_alive(N);

    // Step 5: Define steps and time variables for simulation
    double dt = 1.0;                // timestep in seconds
    double total_step_time = 0.0;   // sum of per-step times
    int steps = Config::NUM_STEPS;
    auto total_start = std::chrono::high_resolution_clock::now();

    // Step 6: Main simulation loop
    for (int step = 0; step < steps; ++step) {
        // Begin timing this one step
        auto t0 = std::chrono::high_resolution_clock::now();

        // Advance system state by one timestep
        // Detect and resolve collisions
        integrate_cuda(d, N, dt);
        check_collisions_cuda(d, N);
        cudaDeviceSynchronize();
 
        // End timing for this step, accumulate total time step, and save time results
        auto t1 = std::chrono::high_resolution_clock::now();
        double step_sec = std::chrono::duration<double>(t1 - t0).count();
        total_step_time += step_sec;
        timefile << step << "," << step_sec << "\n";
 
        // Copy positions + alive flags back to CPU for output
        copyPositionsFromGPU(d, N, host_pos_x, host_pos_y, host_pos_z, host_alive);
        for (int i = 0; i < N; ++i) {
            if (!host_alive[i]) continue;
            outfile << step * dt << "," << i << ","
                    << host_pos_x[i] << "," << host_pos_y[i] << "," << host_pos_z[i] << "\n";
        }
        // Occasionally print state to console (every 100 steps)
        if (step % 100 == 0) {
            int alive_count = 0;
            for (int i = 0; i < N; ++i) alive_count += host_alive[i];
            std::cout << "Step " << step << ": " << alive_count << " bodies alive\n";
        }
    }

    // Step 7: Compute total simulation time
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    double average_step_time = total_step_time / steps;
    std::cout << "Total simulation time: " << total_duration.count() << " seconds\n";
    std::cout << "Average time per step: " << average_step_time << " seconds\n";
    timefile << "total," << total_duration.count() << "\n";
    timefile << "average_per_step," << average_step_time << "\n";
    
    return 0;
}
#endif