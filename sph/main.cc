#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include "opengl.hh"
#include "vector.hh"
#include <GL/glew.h>
#include <GL/glut.h>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using clock_type = std::chrono::high_resolution_clock;
using float_duration = std::chrono::duration<float>;
using vec2 = Vector<float, 2>;

// Original code: https://github.com/cerrno/mueller-sph
constexpr const float kernel_radius = 16;
constexpr const float particle_mass = 65;
constexpr const float poly6 = 315.f / (65.f * float(M_PI) * std::pow(kernel_radius, 9));
constexpr const float spiky_grad = -45.f / (float(M_PI) * std::pow(kernel_radius, 6));
constexpr const float visc_laplacian = 45.f / (float(M_PI) * std::pow(kernel_radius, 6));
constexpr const float gas_const = 2000.f;
constexpr const float rest_density = 1000.f;
constexpr const float visc_const = 250.f;
constexpr const vec2 G(0.f, 12000 * -9.8f);

enum class Version {  CPU,  GPU };
Version version;

struct Particle {

  vec2 position;
  vec2 velocity;
  vec2 force;
  float density;
  float pressure;

  Particle() = default;
  inline explicit Particle(vec2 x) : position(x) {}
};

struct ParticleCL {
  float positionX;
  float positionY;
  float velocityX;
  float velocityY;
  float forceX;
  float forceY;
  float density;
  float pressure;
};

struct OpenCL {
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
};

std::vector<Particle> particles;

OpenCL opencl;
std::vector<float> positions;
cl::Buffer d_positions;
cl::Buffer d_velocities;
cl::Buffer d_forces;
cl::Buffer d_densities;
cl::Buffer d_pressures;

void generate_particles() {
  std::random_device dev;
  std::default_random_engine prng(dev());
  float jitter = 1;
  std::uniform_real_distribution<float> dist_x(-jitter, jitter);
  std::uniform_real_distribution<float> dist_y(-jitter, jitter);
  int ni = 15;
  int nj = 40;
  float x0 = window_width * 0.25f;
  float x1 = window_width * 0.75f;
  float y0 = window_height * 0.20f;
  float y1 = window_height * 1.00f;
  float step = 1.5f * kernel_radius;
  for (float x = x0; x < x1; x += step) {
    for (float y = y0; y < y1; y += step) {
      particles.emplace_back(vec2{x + dist_x(prng), y + dist_y(prng)});
    }
  }
  std::clog << "No. of particles: " << particles.size() << std::endl;
}

void generate_particlesCL() {
  std::random_device dev;
  std::default_random_engine prng(dev());
  float jitter = 1;
  std::uniform_real_distribution<float> dist_x(-jitter, jitter);
  std::uniform_real_distribution<float> dist_y(-jitter, jitter);
  int ni = 15;
  int nj = 40;
  float x0 = window_width * 0.25f;
  float x1 = window_width * 0.75f;
  float y0 = window_height * 0.20f;
  float y1 = window_height * 1.00f;
  float step = 1.5f * kernel_radius;
  for (float x = x0; x < x1; x += step) {
    for (float y = y0; y < y1; y += step) {
      positions.emplace_back(x + dist_x(prng));
      positions.emplace_back(y + dist_y(prng));
    }
  }

  int positions_size = positions.size();
  int count = positions_size / 2;

  opencl.queue.flush();
  d_positions =
      cl::Buffer(opencl.queue, begin(positions), end(positions), true);
  d_velocities = cl::Buffer(opencl.context, CL_MEM_READ_WRITE,
                            positions_size * sizeof(float));
  d_forces = cl::Buffer(opencl.context, CL_MEM_READ_WRITE,
                        positions_size * sizeof(float));
  d_densities =
      cl::Buffer(opencl.context, CL_MEM_READ_WRITE, count * sizeof(float));
  d_pressures =
      cl::Buffer(opencl.context, CL_MEM_READ_WRITE, count * sizeof(float));

  std::clog << "No. of particles: " << count << std::endl;
}

void compute_density_and_pressure() {
  const auto kernel_radius_squared = kernel_radius * kernel_radius;
#pragma omp parallel for schedule(dynamic)
  for (auto &a : particles) {
    float sum = 0;
    for (auto &b : particles) {
      auto sd = square(b.position - a.position);
      if (sd < kernel_radius_squared) {
        sum += particle_mass * poly6 * std::pow(kernel_radius_squared - sd, 3);
      }
    }
    a.density = sum;
    a.pressure = gas_const * (a.density - rest_density);
  }
}

void compute_forces() {
#pragma omp parallel for schedule(dynamic)
  for (auto &a : particles) {
    vec2 pressure_force(0.f, 0.f);
    vec2 viscosity_force(0.f, 0.f);
    for (auto &b : particles) {
      if (&a == &b) {
        continue;
      }
      auto delta = b.position - a.position;
      auto r = length(delta);
      if (r < kernel_radius) {
        pressure_force += -unit(delta) * particle_mass *
                          (a.pressure + b.pressure) / (2.f * b.density) *
                          spiky_grad * std::pow(kernel_radius - r, 2.f);
        viscosity_force += visc_const * particle_mass *
                           (b.velocity - a.velocity) / b.density *
                           visc_laplacian * (kernel_radius - r);
      }
    }
    vec2 gravity_force = G * a.density;
    a.force = pressure_force + viscosity_force + gravity_force;
  }
}

void compute_positions() {
  const float time_step = 0.0008f;
  const float eps = kernel_radius;
  const float damping = -0.5f;
#pragma omp parallel for
  for (auto &p : particles) {
    // forward Euler integration
    p.velocity += time_step * p.force / p.density;
    p.position += time_step * p.velocity;
    // enforce boundary conditions
    if (p.position(0) - eps < 0.0f) {
      p.velocity(0) *= damping;
      p.position(0) = eps;
    }
    if (p.position(0) + eps > window_width) {
      p.velocity(0) *= damping;
      p.position(0) = window_width - eps;
    }
    if (p.position(1) - eps < 0.0f) {
      p.velocity(1) *= damping;
      p.position(1) = eps;
    }
    if (p.position(1) + eps > window_height) {
      p.velocity(1) *= damping;
      p.position(1) = window_height - eps;
    }
  }
}

void on_display() {
  if (no_screen) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  }
  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();
  gluOrtho2D(0, window_width, 0, window_height);
  glColor4f(0.2f, 0.6f, 1.0f, 1);
  glBegin(GL_POINTS);
  switch (version) {
  case Version::CPU: {
    for (const auto &particle : particles) {
      glVertex2f(particle.position(0), particle.position(1));
    }
    break;
  }
  case Version::GPU: {
    for (int i = 0; i < positions.size() / 2; i++) {
      glVertex2f(positions[i * 2], positions[i * 2 + 1]);
    }
    break;
  }
  }
  glEnd();
  glutSwapBuffers();
  if (no_screen) {
    glReadBuffer(GL_RENDERBUFFER);
  }
  recorder.record_frame();
  if (no_screen) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
}

void on_idle_cpu() {
  if (particles.empty()) {
    generate_particles();
  }
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  using std::chrono::seconds;
  auto t0 = clock_type::now();
  compute_density_and_pressure();
  compute_forces();
  compute_positions();
  auto t1 = clock_type::now();
  auto dt = duration_cast<float_duration>(t1 - t0).count();
  std::clog      << std::setw(20) << dt      << std::setw(20) << 1.f / dt      << std::endl;
  glutPostRedisplay();
}

void on_idle_gpu() {
  if (positions.empty()) {
    generate_particlesCL();
  }
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  using std::chrono::seconds;
  auto t0 = clock_type::now();
  int count = positions.size() / 2;

  cl::Kernel kernel_density_and_pressure(opencl.program,
                                         "compute_density_and_pressure");
  kernel_density_and_pressure.setArg(0, d_positions);
  kernel_density_and_pressure.setArg(1, d_densities);
  kernel_density_and_pressure.setArg(2, d_pressures);
  opencl.queue.flush();

  opencl.queue.enqueueNDRangeKernel(kernel_density_and_pressure, cl::NullRange,
                                    cl::NDRange(count), cl::NullRange);
  opencl.queue.flush();

  cl::Kernel kernel_forces(opencl.program, "compute_forces");
  kernel_forces.setArg(0, d_positions);
  kernel_forces.setArg(1, d_velocities);
  kernel_forces.setArg(2, d_densities);
  kernel_forces.setArg(3, d_pressures);
  kernel_forces.setArg(4, d_forces);
  opencl.queue.flush();

  opencl.queue.enqueueNDRangeKernel(kernel_forces, cl::NullRange,
                                    cl::NDRange(count), cl::NullRange);
  opencl.queue.flush();

  cl::Kernel kernel_positions(opencl.program, "compute_positions");
  kernel_positions.setArg(0, d_positions);
  kernel_positions.setArg(1, d_velocities);
  kernel_positions.setArg(2, d_densities);
  kernel_positions.setArg(3, d_pressures);
  kernel_positions.setArg(4, d_forces);
  kernel_positions.setArg(5, window_width);
  kernel_positions.setArg(6, window_height);
  opencl.queue.flush();

  opencl.queue.enqueueNDRangeKernel(kernel_positions, cl::NullRange,
                                    cl::NDRange(count), cl::NullRange);
  opencl.queue.flush();

  opencl.queue.enqueueReadBuffer(
      d_positions, true, 0, positions.size() * sizeof(float), positions.data());
  opencl.queue.flush();

  auto t1 = clock_type::now();
  auto dt = duration_cast<float_duration>(t1 - t0).count();
  std::clog      << std::setw(20) << dt      << std::setw(20) << 1.f / dt      << std::endl;
  glutPostRedisplay();
}

void on_keyboard(unsigned char c, int x, int y) {
  switch (c) {
  case ' ':
    generate_particles();
    break;
  case 'r':
  case 'R':
    particles.clear();
    generate_particles();
    break;
  }
}

void print_column_names() {
  std::clog << std::setw(20) << "Frame duration";
  std::clog << std::setw(20) << "Frames per second";
  std::clog << '\n';
}

const std::string src = R"(
#define kernel_radius 16.f
#define particle_mass 65.f
#define poly6 315.f/(65.f*(float)(M_PI)*pow(kernel_radius,9.f))
#define spiky_grad -45.f/((float)(M_PI)*pow(kernel_radius,6.f))
#define visc_laplacian 45.f/((float)(M_PI)*pow(kernel_radius,6.f))
#define gas_const 2000.f
#define rest_density 1000.f
#define visc_const 250.f
#define G (float2)(0.f, 12000*-9.8f)

kernel void compute_density_and_pressure(global const float* positions,
                        global float* densities,
                        global float* pressures) {

    float kernel_radius_squared = kernel_radius*kernel_radius;
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);

    float sum = 0;
    float aX = positions[global_id*2];
    float aY = positions[global_id*2+1];
    for (int i = 0; i < global_size; ++i) {
        float bX = positions[i*2];
        float bY = positions[i*2+1];
        float sd = (bX - aX)*(bX - aX) + (bY - aY)*(bY - aY);
        if (sd < kernel_radius_squared) {
            sum += particle_mass*poly6*pow(kernel_radius_squared-sd, 3);
        }
    }
    densities[global_id] = sum;
    pressures[global_id] = gas_const*(sum - rest_density);
}

kernel void compute_forces(global const float* positions,
                        global const float* velocities,
                        global const float* densities,
                        global const float* pressures,
                        global float* forces) {    
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float2 pressure_force = (float2)(0.f, 0.f);
    float2 viscosity_force = (float2)(0.f, 0.f);
    float2 a_position = (float2)(positions[global_id*2], positions[global_id*2+1]);
    float2 a_velocity = (float2)(velocities[global_id*2], velocities[global_id*2+1]);
    float a_pressure = pressures[global_id];    
    float a_density = densities[global_id];    
    for (int i = 0; i < global_size; ++i) {
        if (global_id == i) { continue; }
        float2 b_position = (float2)(positions[i*2], positions[i*2+1]);
        float2 b_velocity = (float2)(velocities[i*2], velocities[i*2+1]);
        float b_pressure = pressures[i];
        float b_density = densities[i];
        float2 delta = b_position - a_position;
        float r = length(delta);
        if (r < kernel_radius) {
            pressure_force += -normalize(delta)*particle_mass*(a_pressure + b_pressure)
                / (2.f * b_density)
                * spiky_grad*pow(kernel_radius-r,2.f);
            viscosity_force += visc_const*particle_mass*(b_velocity - a_velocity)
                / b_density * visc_laplacian*(kernel_radius-r);
        }
    }
    float2 gravity_force = G * a_density;
    float2 f = pressure_force + viscosity_force + gravity_force;
    forces[global_id*2] = f[0];
    forces[global_id*2+1] = f[1]; 
}

kernel void compute_positions(global float* positions,
                        global float* velocities,
                        global const float* densities,
                        global const float* pressures,
                        global const float* forces,
                        int window_width,
                        int window_height) {
    float time_step = 0.0008f;
    float eps = kernel_radius;
    float damping = -0.5f;
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);

    // forward Euler integration
    float2 force = (float2)(forces[global_id*2], forces[global_id*2+1]);
    float2 velocity = (float2)(velocities[global_id*2], velocities[global_id*2+1]);
    float density = densities[global_id];

    float2 velocity_incr = time_step*force/density;
    velocities[global_id*2] += velocity_incr[0];
    velocities[global_id*2+1] += velocity_incr[1];

    float2 velocity_updated = (float2)(velocities[global_id*2], velocities[global_id*2+1]);
    float2 position_incr = time_step*velocity_updated;
    positions[global_id*2] += position_incr[0];
    positions[global_id*2+1] += position_incr[1];

    // enforce boundary conditions
    if (positions[global_id*2]-eps < 0.0f) {
        velocities[global_id*2] *= damping;
        positions[global_id*2] = eps;
    }
    if (positions[global_id*2]+eps > window_width) {
      velocities[global_id*2] *= damping;
      positions[global_id*2] = window_width-eps;
    }
    if (positions[global_id*2+1]-eps < 0.0f) {
      velocities[global_id*2+1] *= damping;
      positions[global_id*2+1] = eps;
    }
    if (positions[global_id*2+1]+eps > window_height) {
      velocities[global_id*2+1] *= damping;
      positions[global_id*2+1] = window_height-eps;
    }
}

)";

void newOpenCL() {
  try {
    // find OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
      std::cerr << "Unable to find OpenCL platforms\n";
      return;
    }
    cl::Platform platform = platforms[0];
    std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>()
              << '\n';
    // create context
    cl_context_properties properties[] =        {CL_CONTEXT_PLATFORM,
                                          (cl_context_properties)platform(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);
    // get all devices associated with the context
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices[0];
    std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
    cl::Program program(context, src);
    // compile the programme
    try {
      program.build(devices);
    } catch (const cl::Error &err) {
      for (const auto &device : devices) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << log;
      }
      throw;
    }
    cl::CommandQueue queue(context, device);
    opencl = {platform, device, context, program, queue};
  } catch (const cl::Error &err) {
    std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
    std::cerr << "Search cl.h file for error code (" << err.err()
              << ") to understand what it means:\n";
    std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/"
                 "CL/cl.h\n";
    return;
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    return;
  }
}

int main(int argc, char *argv[]) {
  version = Version::CPU;
  if (argc == 2) {
    std::string str(argv[1]);
    for (auto &ch : str) {
      ch = std::tolower(ch);
    }
    if (str == "gpu") {
      version = Version::GPU;
    }
  }
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
  glutInitWindowSize(window_width, window_height);
  glutInit(&argc, argv);
  glutCreateWindow("SPH");
  glutDisplayFunc(on_display);
  glutReshapeFunc(on_reshape);
  switch (version) {
  case Version::CPU:
    glutIdleFunc(on_idle_cpu);
    break;
  case Version::GPU: {
    newOpenCL();
    glutIdleFunc(on_idle_gpu);
    break;
  }
  default:
    return 1;
  }
  glutKeyboardFunc(on_keyboard);
  glewInit();
  init_opengl(kernel_radius);
  print_column_names();
  glutMainLoop();
  return 0;
}