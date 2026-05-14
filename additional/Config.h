// Config.h
#ifndef CONFIG_H
#define CONFIG_H

// ++++++++ IMPORTS ++++++++ //
#include <cmath>
#include <iostream>


// ++++++++ STRUCTS ++++++++ //
struct Vec3 {
    // 3D Vector Struct
    // Includes operations for + - * / and norm
    double x, y, z;
    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s) const { return {x/s, y/s, z/s}; }
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

struct Body {
    // Body Struct
    Vec3 pos;
    Vec3 vel;
    Vec3 acc;
    double mass;
    double radius;
    bool alive = true; // for collisions
};

struct BodyGPU {
    // Body Struct specifically for optimized GPU memory layout
    // Why: GPU doesn't run one thread at a time, runs 32 threads simultaneously in a warp
    // 32 threads execute the same instruction at the same time, just on different data
    // Option 1: struct of arrays access pattern is best:
        // BodyGPU holds 12 separate flat arrays from allocBodyGPU in run_cuda.cu
        // pos_x: [x0,  x1,  x2,  x3,  x4  ... xN]
        // thread 0 reads pos_x[0]
        // thread 1 reads pos_x[1] <= immediately adjacent
    // Option 2: array of structs access pattern is bad:
        // from the serial version, std::vector<Body> would appear like:
        // [x,y,z, vx,vy,vz, ...,  x,y,z, vx,vy,vz, ...]
        // thread 0 reads address 0 to get x
        // thread 1 reads address 24 to get next x (skipping y, z, etc., not adjacent)
    double *pos_x, *pos_y, *pos_z;
    double *vel_x, *vel_y, *vel_z;
    double *acc_x, *acc_y, *acc_z;
    double *mass;
    double *radius;
    int *alive;
};


// ++++++++ CLASSES ++++++++ //
class Config {
    // Global configuration class
    public:
        // Planet parameters
        static constexpr double G = 6.67430e-11;         // gravitational constant
        static constexpr double planet_mass = 5.972e24;  // kg, Earth-like
        static constexpr double planet_radius = 6.371e6; // meters
    
        // Body parameters
        static constexpr int NUM_BODIES = 5000;
        static constexpr double MIN_ORBIT_RADIUS = 7e6; // 700 km above surface
        static constexpr double MAX_ORBIT_RADIUS = 15e6; // 1500 km above surface
        static constexpr int BODY_MASS = 1000;
        static constexpr int BODY_RADIUS = 1;
        static constexpr int NUM_STEPS = 50;
};

#endif