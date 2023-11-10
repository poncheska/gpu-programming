#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

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

#include "color.hh"
#include "random.hh"
#include "ray.hh"
#include "scene.hh"
#include "theora.hh"
#include "vector.hh"

using uniform_distribution = std::uniform_real_distribution<float>;
using color = Color<float>;
using object_ptr = std::unique_ptr<Object>;
using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

vec trace(ray r, const Object_group &objects) {
  float factor = 1;
  const int max_depth = 50;
  int depth = 0;
  for (; depth < max_depth; ++depth) {
    if (Hit hit = objects.hit(r, 1e-3f, std::numeric_limits<float>::max())) {
      r = ray(hit.point, hit.normal + random_in_unit_sphere()); // scatter
      factor *= 0.5f; // diffuse 50% of light, scatter the remaining
    } else {
      break;
    }
  }
  // if (depth == max_depth) { return vec{}; }
  //  nothing was hit
  //  represent sky as linear gradient in Y dimension
  float t = 0.5f * (unit(r.direction())(1) + 1.0f);
  return factor *
         ((1.0f - t) * vec(1.0f, 1.0f, 1.0f) + t * vec(0.5f, 0.7f, 1.0f));
}

struct OpenCL {
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
};

void print_column_names(const char *version) {
  std::cout << std::setw(20) << "Time step";
  std::cout << std::setw(20) << "No. of steps";
  std::cout << std::setw(20) << version << " time";
  std::cout << '\n';
}

void ray_tracing_cpu() {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  using std::chrono::seconds;
  int nx = 600, ny = 400, nrays = 100;
  Pixel_matrix<float> pixels(nx, ny);
  thx::screen_recorder recorder("out.ogv", nx, ny);
  Object_group objects;
  objects.add(object_ptr(new Sphere(vec(0.f, 0.f, -1.f), 0.5f)));
  objects.add(object_ptr(new Sphere(vec(0.f, -1000.5f, -1.f), 1000.f)));
  Camera camera;
  uniform_distribution distribution(0.f, 1.f);
  float gamma = 2;
  const int max_time_step = 60;
  print_column_names("OpenMP");
  duration total_time = duration::zero();
  for (int time_step = 1; time_step <= max_time_step; ++time_step) {
    auto t0 = clock_type::now();
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        vec sum;
        for (int k = 0; k < nrays; ++k) {
          float u = (i + distribution(prng)) / nx;
          float v = (j + distribution(prng)) / ny;
          sum += trace(camera.make_ray(u, v), objects);
        }
        sum /= float(nrays);         // antialiasing
        sum = pow(sum, 1.f / gamma); // gamma correction
        pixels(i, j) = to_color(sum);
      }
    }
    auto t1 = clock_type::now();
    const auto dt = duration_cast<microseconds>(t1 - t0);
    total_time += dt;
    std::clog        << std::setw(20) << time_step        << std::setw(20) << max_time_step
          << std::setw(20) << dt.count()        << std::endl;
    std::ofstream out("out.ppm");
    out << pixels;
    recorder.record_frame(pixels);
    camera.move(vec{0.f, 0.f, 0.1f});
  }
  std::clog << "Ray-tracing time: "
            << duration_cast<seconds>(total_time).count()            << "s" << std::endl;
  std::clog << "Movie time: " << max_time_step / 60.f << "s" << std::endl;
}

void ray_tracing_gpu(OpenCL &opencl) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  using std::chrono::seconds;
  int nx = 600, ny = 400, nrays = 100;
  int result_size = nx * ny * 3;
  Pixel_matrix<float> pixels(nx, ny);
  thx::screen_recorder recorder("out.ogv", nx, ny);
  Sphere s1 = Sphere{vec(0.f, 0.f, -1.f), 0.5f};
  Sphere s2 = Sphere{vec(0.f, -1000.5f, -1.f), 1000.f};

  std::vector<float> s1_vec;
  s1_vec.emplace_back(s1.origin()(0));
  s1_vec.emplace_back(s1.origin()(1));
  s1_vec.emplace_back(s1.origin()(2));
  s1_vec.emplace_back(s1.radius());

  std::vector<float> s2_vec;
  s2_vec.emplace_back(s2.origin()(0));
  s2_vec.emplace_back(s2.origin()(1));
  s2_vec.emplace_back(s2.origin()(2));
  s2_vec.emplace_back(s2.radius());

  Camera camera;

  uniform_distribution uni_distribution(0.f, 1.f);
  std::vector<float> distr_for_ray;
  for (int i = 0; i < nrays * 2; ++i) {
    distr_for_ray.emplace_back(uni_distribution(prng));
  }

  std::uniform_real_distribution<float> real_distribution(-1, 1);
  std::vector<float> distr_for_sphere;
  for (int i = 0; i < nx * ny * nrays; ++i) {
    for (int j = 0; j < 4; ++j) {
      distr_for_sphere.emplace_back(real_distribution(prng));
    }
  }

  float gamma = 2;
  const int max_time_step = 60;
  print_column_names("OpenCL");
  duration total_time = duration::zero();
  opencl.queue.flush();
  cl::Kernel kernel(opencl.program, "trace_ray");
  cl::Buffer d_distr_ray(opencl.queue, begin(distr_for_ray), end(distr_for_ray),
                         true);
  cl::Buffer d_distr_sphere(opencl.queue, begin(distr_for_sphere),
                            end(distr_for_sphere), true);
  cl::Buffer d_s1_vec(opencl.queue, begin(s1_vec), end(s1_vec), true);
  cl::Buffer d_s2_vec(opencl.queue, begin(s2_vec), end(s2_vec), true);
  cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE,
                      result_size * sizeof(float));
  kernel.setArg(0, d_distr_ray);
  kernel.setArg(1, d_distr_sphere);
  kernel.setArg(2, d_s1_vec);
  kernel.setArg(3, d_s2_vec);
  kernel.setArg(4, nrays);
  kernel.setArg(5, gamma);
  kernel.setArg(6, nx);
  kernel.setArg(7, ny);
  kernel.setArg(8, d_result);

  for (int time_step = 1; time_step <= max_time_step; ++time_step) {
    auto t0 = clock_type::now();
    std::vector<float> camera_origin_vec;
    camera_origin_vec.emplace_back(camera.origin()(0));
    camera_origin_vec.emplace_back(camera.origin()(1));
    camera_origin_vec.emplace_back(camera.origin()(2));
    cl::Buffer d_camera_origin_vec(opencl.queue, begin(camera_origin_vec),
                                   end(camera_origin_vec), true);
    kernel.setArg(9, d_camera_origin_vec);
    opencl.queue.flush();

    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(nx, ny), cl::NullRange);
    opencl.queue.flush();

    opencl.queue.enqueueReadBuffer(d_result, true, 0,
                                   result_size * sizeof(float),
                                   (float *)(pixels.pixels().data()));
    opencl.queue.flush();

    auto t1 = clock_type::now();
    const auto dt = duration_cast<microseconds>(t1 - t0);
    total_time += dt;
    std::clog        << std::setw(20) << time_step        << std::setw(20) << max_time_step
          << std::setw(20) << dt.count()        << std::endl;
    std::ofstream out("out.ppm");
    out << pixels;
    recorder.record_frame(pixels);
    camera.move(vec{0.f, 0.f, 0.1f});
  }
  std::clog << "Ray-tracing time: "
            << duration_cast<seconds>(total_time).count()            << "s" << std::endl;
  std::clog << "Movie time: " << max_time_step / 60.f << "s" << std::endl;
}

const std::string src = R"(
typedef struct {
    float3 origin;
    float3 direction;
} ray;

typedef struct {
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    float3 origin;
} camera;

typedef struct {
    float3 origin;
    float radius;
} sphere;

typedef struct {
    float t;
    float3 point;
    float3 normal;
} Hit;

float3 random_in_unit_sphere(const global float* dist, int k, int nrays){
    float4 x;
    float square_x;
    int i = 0;
    int dist_len = nrays * 4;
    for (int i = 0; i < dist_len; ++i) {
        float x0 = dist[(k+i)%dist_len];
        float x1 = dist[(k+i+1)%dist_len];
        float x2 = dist[(k+i+2)%dist_len];
        float x3 = dist[(k+i+3)%dist_len];
        x = (float4)(x0, x1, x2, x3);
        square_x = x0*x0 + x1*x1 + x2*x2 + x3*x3;
        if (square_x < 1.f) {
            break;
        }
    }
    // quaternions!
    float x_f = 2*(x.y*x.z+x.x*x.z);
    float y = 2*(x.z*x.w-x.x*x.y);
    float z = x.x*x.x+x.w*x.w-x.y*x.y-x.z*x.z;
    float3 result = (float3)(x_f, y, z);
    return result / square_x;
    //return (float3)(0.5f, 0.5f, 0.5f);
}

Hit hit_object(ray r, float t_min, float t_max, sphere s){
    Hit result;
    result.t = -1.f;
    float3 oc = r.origin - s.origin;
    float a = dot(r.direction, r.direction);
    float b = dot(oc, r.direction);
    float c = dot(oc, oc) - s.radius*s.radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float d = sqrt(discriminant);
        float t = (-b - d)/a;
        bool success = false;
        if (t_min < t && t < t_max) {
            success = true;
        } else {
            t = (-b + d)/a;
            if (t_min < t && t < t_max) { success = true; }
        }
        if (success) {
            result.t = t;
            result.point = r.origin + t*r.direction;
            result.normal = (result.point - s.origin) / s.radius;
        }
    }
    return result;
}

Hit hit_objects(ray r, float t_min, float t_max, sphere s1, sphere s2) {
    Hit result;
    result.t = -1.f;
    Hit h1 = hit_object(r, t_min, t_max, s1);
    if (h1.t > 0) {
        result = h1;
        t_max = h1.t;
    }
    Hit h2 = hit_object(r, t_min, t_max, s2);
    if (h2.t > 0) {
        result = h2;
        t_max = h2.t;
    }
    return result;
}

float3 trace(ray r, const global float* dist, int k, int nrays, sphere s1, sphere s2) {
    float factor = 1;
    const int max_depth = 50;
    int depth=0;
    for (; depth<max_depth; ++depth) {
        Hit hit = hit_objects(r, 1e-3f, FLT_MAX, s1, s2);
        if (hit.t > 0) {
            r.origin = hit.point;
            r.direction = hit.normal + random_in_unit_sphere(dist, k, nrays);
            factor *= 0.5f; // diffuse 50% of light, scatter the remaining
        } else {
          break;
        }
    }
    //if (depth == max_depth) { return vec{}; }
    // nothing was hit
    // represent sky as linear gradient in Y dimension
    float t = 0.5f*(r.direction.y + 1.0f);
    return factor*((1.0f-t)*(float3)(1.0f, 1.0f, 1.0f) + t*(float3)(0.5f, 0.7f, 1.0f));
}

ray camera_make_ray(float u,
                    float v,
                    camera cam) {
  ray r;
  r.origin = cam.origin;
  r.direction = cam.lower_left_corner + u*cam.horizontal + v*cam.vertical - cam.origin;
  return r;
}

kernel void trace_ray(  global const float* distribution_for_ray,
                      global const float* distribution_for_sphere,
                      global const float* s1_vec,
                      global const float* s2_vec,
                      int nrays,
                      float gamma,
                      int nx,
                      int ny,
                      global float* result,
                      global const float* camera_origin) {
  sphere s1;
  s1.origin = (float3)(s1_vec[0], s1_vec[1], s1_vec[2]);
  s1.radius = s1_vec[3];

  sphere s2;
  s2.origin = (float3)(s2_vec[0], s2_vec[1], s2_vec[2]);
  s2.radius = s2_vec[3];

  camera cam;
  cam.lower_left_corner = (float3)(3.02374f,-1.22628f,3.4122f);
  cam.horizontal = (float3)(1.18946f,0.f,-5.15434f);
  cam.vertical = (float3)(-0.509421f,3.48757f,-0.117559f);
  cam.origin = (float3)(camera_origin[0],camera_origin[1],camera_origin[2]);

  int i = get_global_id(0);
  int j = get_global_id(1);

  float3 sum = (float3)(0.f, 0.f, 0.f);
  for (int k=0; k<nrays; ++k) {
    float u = (i + distribution_for_ray[k*2]) / nx;
    float v = (j + distribution_for_ray[k*2+1]) / ny;
    ray r = camera_make_ray(u, v, cam);
    sum += trace(r, distribution_for_sphere, k, nrays, s1, s2);
  }
  sum /= (float)(nrays); // antialiasing
  sum = pow(sum,1.f/gamma); // gamma correction

  int index = j*nx + i;
  result[3 * index] = sum.x;
  result[3 * index +1] = sum.y;
  result[3 * index +2] = sum.z;
}

)";

int main(int argc, char *argv[]) {
  enum class Version {    CPU,    GPU  };
  Version version = Version::CPU;
  if (argc == 2) {
    std::string str(argv[1]);
    for (auto &ch : str) {
      ch = std::tolower(ch);
    }
    if (str == "gpu") {
      version = Version::GPU;
    }
  }
  switch (version) {
  case Version::CPU:
    ray_tracing_cpu();
    break;
  case Version::GPU: {
    try {
      // find OpenCL platforms
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      if (platforms.empty()) {
        std::cerr << "Unable to find OpenCL platforms\n";
        return 1;
      }
      cl::Platform platform = platforms[0];
      std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>()
                << '\n';
      // create context
      cl_context_properties properties[] =          {
          CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
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
      OpenCL opencl{platform, device, context, program, queue};
      ray_tracing_gpu(opencl);
    } catch (const cl::Error &err) {
      std::cerr << "OpenCL error in " << err.what() << '(' << err.err()
                << ")\n";
      std::cerr << "Search cl.h file for error code (" << err.err()
                << ") to understand what it means:\n";
      std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/"
                   "CL/cl.h\n";
      return 1;
    } catch (const std::exception &err) {
      std::cerr << err.what() << std::endl;
      return 1;
    }
    break;
  }
  default:
    return 1;
  }
  return 0;
}