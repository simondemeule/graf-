/**
 @file
 graf~.cpp
 
 @description
 <o>graf~</o> is an experimental GPU audio max external
 
 @author
 Simon Demeule
 
 @owner
 Simon Demeule
 */

#include "ext.h"            // standard Max include, always required (except in Jitter)
#include "ext_obex.h"        // required for "new" style objects
#include "ext_buffer.h"
#include "z_dsp.h"            // required for MSP objects

#define CL_SILENCE_DEPRECATION

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include <sys/time.h>

// struct to represent the object's state
typedef struct _graf {
    t_pxobject ob;            // the object itself (t_pxobject in MSP instead of t_object)
    double samplerate;
    long sampleframes;
    
    bool cl_init_attempted = false;
    bool cl_init_failed = false;
    size_t cl_global;                      // global domain size for our calculation
    size_t cl_local;                       // local domain size for our calculation
    cl_int cl_err;
    int cl_device_idx;
    cl_device_id cl_device_id;             // compute device id
    cl_context cl_context;                 // compute context
    cl_command_queue cl_queue;          // compute command queue
    cl_program cl_program;                 // compute program
    cl_kernel cl_kernel;                   // compute kernel
} t_graf;

// method prototypes
void *graf_new(t_symbol *s, long argc, t_atom *argv);
void graf_free(t_graf *x);
void graf_dsp64(t_graf *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);

void graf_float(t_graf *x, double f);
void graf_int(t_graf *x, long l);

void graf_assist(t_graf *x, void *b, long m, long a, char *s);
void graf_set(t_graf *x, t_symbol *s);

void graf_perform64(t_graf *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);

// global class pointer variable
static t_class *graf_class = NULL;

void ext_main(void *r)
{
    t_class *c = class_new("graf~", (method)graf_new, (method)graf_free, (long)sizeof(t_graf), 0L, A_GIMME, 0);
    
    class_addmethod(c, (method)graf_dsp64,          "dsp64",        A_CANT,     0);
    class_addmethod(c, (method)graf_float,          "float",        A_FLOAT,    0);
    class_addmethod(c, (method)graf_int,            "int",          A_LONG,     0);
    class_addmethod(c, (method)graf_assist,         "assist",       A_CANT,     0);
    
    class_dspinit(c);
    class_register(CLASS_BOX, c);
    graf_class = c;
}

// THE FFT HAS 2 CHANNELS. WHAT DO WE DO WITH THAT???? :(:(:(:(:(:(:(

// buffer(bin, time) is a data structure containing previous output values + the current input value
// bin  spans (0, bin_size - 1)
// time spans (0, time_size - 1)
//
// buffer(0, 0)              • • •  buffer(bin_size - 1, 0)
//   •                                •
//   •                                •
//   •                                •
// buffer(0, time_size - 1)  • • •  buffer(bin_size - 1, time_size - 1)
//
// structure works with rolling buffer over time
// buffer time_offset is given by time_offset
// current input frame exists at buffer(0, time_offset) • • • buffer(bin_size, time_offset)
// previous input frame exists at buffer(0, warp(time_offset - 1, time_size)) • • • buffer(N, warp(time_offset - 1, time_size))
// where warp(i, time_size) is non-negative modulo M
//
// this 2D array is flattened with
// buffer_flat(index) = buffer(warp(index, bin_size), index / bin_size)
//
// conversely we have
// buffer(bin, time) = buffer_flat(bin + time * bin_size)
//
// which gives us the map
//
// buffer_flat(0)  • • •  buffer_flat(bin_size - 1), buffer_flat(bin_size)  • • •  buffer_flat(bin_size * time_size - 1)
//
// buffer(0, 0)    • • •  buffer(bin_size - 1, 0),   buffer(0, 1)           • • •  buffer(bin_size - 1, time_size - 1)
//
// the linear system is defined as
//
// output(bin_output) = sum(bin_input from 0 to bin_size - 1,
//                      sum(time from 0 to time_size - 1,
//                      buffer(bin_input, warp(time_offset - time, time_size)) * coefficient(bin_input, time, bin_output)))
//
// once output(bin_output) is computed for all bin_output, time_offset is set to warp(time_offset + 1, time_size), and buffer(x, warp(T - 1)) is set to o(x) for all x
//
// coefficient(bin_input, time, bin_output) is a data structure containing the system coefficients
// bin_input  spans (0, bin_size - 1)
// time       spans (0, time_size - 1)
// bin_output spans (0, bin_size - 1)
//
//        coefficient(0, 0, bin_size - 1)   • • •  coefficient(bin_size - 1, 0, bin_size - 1)
//         •                                        •
//       •                                        •
// coefficient(0, 0, 0)              • • •  coefficient(bin_size - 1, 0, 0)
//    •                                        •
//    •                                        •
//    •                                        •
// coefficient(0, time_size - 1, 0)  • • •  coefficient(bin_size - 1, time_size - 1, 0)
//
// this 3D array is flattened with
// coefficient_flat(index) = coefficient(warp(index, bin_size), warp(index / bin_size, time_size), index / (bin_size * time_size))
//
// conversely we have
// coefficient(bin_input, time, bin_output) = coefficient_flat(bin_input + time * bin_size + bin_output * bin_size * time_size)
//
// which gives us the map
//
// coefficient_flat(0)   • • •  coefficient_flat(bin_size - 1),  coefficient_flat(bin_size)  • • •  coefficient_flat(bin_size * time_size - 1),  coefficient_flat(bin_size * time_size)  • • •  coefficient_flat(bin_size * time_size * bin_size - 1)
//
// coefficient(0, 0, 0)  • • •  coefficient(bin_size - 1, 0, 0), coefficient(0, 1, 0)        • • •  coefficient(bin_size - 1, time_size - 1, 0), coefficient(0, 0, 1)                    • • •  coefficient(bin_size - 1, time_size - 1, bin_size - 1)
//
// kernels can only use single-dimensional arrays, so they are always implicitly the flattened versions of the array


const char *kernel_source = "\n" \
"int warp(int index, int size) {                                                        \n" \
"   int temp = index % size;                                                            \n" \
"   return temp < 0 ? temp + size : temp;                                               \n" \
"}                                                                                      \n" \
"__kernel void system(                                                                  \n" \
"   __global double* buffer,                                                            \n" \
"   __global double* coefficients,                                                      \n" \
"   __global double* output,                                                            \n" \
"   const long bin_size,                                                                \n" \
"   const long time_size,                                                               \n" \
"   const long time_offset)                                                             \n" \
"{                                                                                      \n" \
"   int bin_output = get_global_id(0);                                                  \n" \
"   if(bin_output < bin_size) {                                                         \n" \
"      double accum = 0;                                                                \n" \
"      for(int time = 0; time < time_size; time++) {                                    \n" \
"         int time_rolling = warp(time_offset - time, time_size);                       \n" \
"         for(int bin_input = 0; bin_input < bin_size; bin_input++) {                   \n" \
"            int n = bin_input + time_rolling * bin_size;                               \n" \
"            int m = bin_input + time * bin_size + bin_output * bin_size * time_size;   \n" \
"            accum += buffer[n] * coefficients[m];                                      \n" \
"         }                                                                             \n" \
"      }                                                                                \n" \
"      output[bin_output] = accum;                                                      \n" \
"   }                                                                                   \n" \
"}                                                                                      \n" \
"\n";

void graf_cl_init(t_graf *x) {
    cl_device_id device_ids[8];
    cl_uint device_ids_length;
    
    x->cl_err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 8, device_ids, &device_ids_length);
    if (x->cl_err != CL_SUCCESS)
    {
        post("Error: Failed to create a device group!\n");
        x->cl_init_failed = true;
        return;
    }
    
    post("Found %i compute devices:\n", device_ids_length);
    for(int i = 0; i < device_ids_length; i++) {
        
        char device_name[128];
        size_t device_name_length;
        
        x->cl_err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, 128, &device_name, &device_name_length);
        if (x->cl_err != CL_SUCCESS)
        {
            post("Error: Failed to retrieve device name!\n");
            x->cl_init_failed = true;
            return;
        }
        
        post("- %s\n", device_name);
    }
    
    // hardcoded device selection
    
#define DEVICE_VEGA20 (2)
#define DEVICE_UHD630 (1)
#define DEVICE_I9 (0)
    
    x->cl_device_id = device_ids[DEVICE_UHD630];
    
    // Vega only has an edge when dimensionality goes above ~2^11
    
    char device_name[128];
    size_t device_name_length;
    
    x->cl_err = clGetDeviceInfo(x->cl_device_id, CL_DEVICE_NAME, 128, &device_name, &device_name_length);
    if (x->cl_err != CL_SUCCESS)
    {
        post("Error: Failed to retrieve device name!\n");
        x->cl_init_failed = true;
        return;
    }
    
    post("Device chosen for computation:\n- %s\n", device_name);
    
    // Create a compute context
    //
    x->cl_context = clCreateContext(0, 1, &(x->cl_device_id), NULL, NULL, &(x->cl_err));
    //context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
    if (!x->cl_context)
    {
        post("Error: Failed to create a compute context!\n");
        x->cl_init_failed = true;
        return;
    }
    
    // Create a command queue
    //
    x->cl_queue = clCreateCommandQueue(x->cl_context, x->cl_device_id, 0, &(x->cl_err));
    if (!x->cl_queue)
    {
        post("Error: Failed to create a command queue!\n");
        x->cl_init_failed = true;
        return;
    }
    
    // Create the compute program from the source buffer
    //
    x->cl_program = clCreateProgramWithSource(x->cl_context, 1, (const char **) &kernel_source, NULL, &(x->cl_err));
    if (!x->cl_program)
    {
        post("Error: Failed to create compute program!\n");
        x->cl_init_failed = true;
        return;
    }
    
    // Build the program executable
    //
    x->cl_err = clBuildProgram(x->cl_program, 0, NULL, NULL, NULL, NULL);
    if (x->cl_err != CL_SUCCESS)
    {
        size_t length;
        char buffer[2048];
        
        post("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(x->cl_program, x->cl_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        post("%s\n", buffer);
        x->cl_init_failed = true;
        return;
    }
    
    // Create the compute kernel in the program we wish to run
    //
    x->cl_kernel = clCreateKernel(x->cl_program, "system", &(x->cl_err));
    if (!x->cl_kernel || x->cl_err != CL_SUCCESS)
    {
        post("Error: Failed to create compute kernel!\n");
        x->cl_init_failed = true;
        return;
    }
    
    // Get the maximum work group size for executing the kernel on the device
    //
    x->cl_err = clGetKernelWorkGroupInfo(x->cl_kernel, x->cl_device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(x->cl_local), &(x->cl_local), NULL);
    if (x->cl_err != CL_SUCCESS)
    {
        post("Error: Failed to retrieve kernel work group info! %d\n", x->cl_err);
        x->cl_init_failed = true;
        return;
    }
    
    x->cl_init_failed = false;
}

void graf_cl_reset(t_graf *x) {
    clReleaseProgram(x->cl_program);
    clReleaseKernel(x->cl_kernel);
    clReleaseCommandQueue(x->cl_queue);
    clReleaseContext(x->cl_context);
}

void *graf_new(t_symbol *s, long argc, t_atom *argv)
{
    t_graf *x = (t_graf *)object_alloc(graf_class);
    
    if (x) {
        dsp_setup((t_pxobject *)x, 2);
        outlet_new(x, "signal");
        // disable in-place optimisation so that the input and output buffers are distinct
        x->ob.z_misc |= Z_NO_INPLACE;
    }
    return (x);
}

void graf_free(t_graf *x)
{
    dsp_free((t_pxobject *)x);
    graf_cl_reset(x);
}

// registers a function for the signal chain in Max
void graf_dsp64(t_graf *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
    post("my sample rate is: %f", samplerate);
    
    x->samplerate = samplerate;
    object_method(dsp64, gensym("dsp_add64"), x, graf_perform64, 0, NULL);
}

//***********************************************************************************************

void graf_float(t_graf *x, double f) {
    long in = proxy_getinlet((t_object *)x);
    
    post("message (float) %f from inlet %d", f, in);
}

void graf_int(t_graf *x, long l) {
    long in = proxy_getinlet((t_object *)x);
    
    post("message (long) %l from inlet %d", l, in);
}

void graf_assist(t_graf *x, void *b, long m, long a, char *s)
{
    if (m == ASSIST_INLET) { //inlet
        switch (a) {
            default:
                sprintf(s, "inlet");
                break;
        }
    }
    else {    // outlet
        sprintf(s, "outlet");
    }
}

// this is the 64-bit perform method audio vectors
void graf_perform64(t_graf *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
    t_double *in = ins[0];
    t_double *out = outs[0];
    
    // possible edge case: change of sampleframes? is the graph rebuilt then?
    
    if(!x->cl_init_attempted) {
        graf_cl_init(x);
        x->cl_init_attempted = true;
    }
    if(!x->cl_init_failed)
    {
        cl_mem cl_mem_input = clCreateBuffer(x->cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * sampleframes, in, &(x->cl_err));
        if (x->cl_err != CL_SUCCESS)
        {
            post("Error: Failed to attach input memory! %d\n", x->cl_err);
        }
        cl_mem cl_mem_output = clCreateBuffer(x->cl_context, CL_MEM_WRITE_ONLY, sizeof(double) * sampleframes, NULL, &(x->cl_err));
        if (x->cl_err != CL_SUCCESS)
        {
            post("Error: Failed to attach output memory! %d\n", x->cl_err);
        }
        
        // Set the arguments to our compute kernel
        //
        x->cl_err  = 0;
        x->cl_err  = clSetKernelArg(x->cl_kernel, 0, sizeof(cl_mem), &cl_mem_input);
        x->cl_err |= clSetKernelArg(x->cl_kernel, 1, sizeof(cl_mem), &cl_mem_output);
        x->cl_err |= clSetKernelArg(x->cl_kernel, 2, sizeof(long), &sampleframes);
        if (x->cl_err != CL_SUCCESS)
        {
            post("Error: Failed to set kernel arguments! %d\n", x->cl_err);
        }
        
        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //
        x->cl_global = sampleframes;
        x->cl_err = clEnqueueNDRangeKernel(x->cl_queue, x->cl_kernel, 1, NULL, &(x->cl_global), &(x->cl_local), 0, NULL, NULL);
        if (x->cl_err)
        {
            post("Error: Failed to execute kernel!\n");
        }
        
        // Wait for the command commands to get serviced before reading back results
        //
        clFinish(x->cl_queue);
        
        // Read back the results from the device to verify the output
        //
        x->cl_err = clEnqueueReadBuffer(x->cl_queue, cl_mem_output, CL_TRUE, 0, sizeof(double) * sampleframes, out, 0, NULL, NULL );
        if (x->cl_err != CL_SUCCESS)
        {
            post("Error: Failed to read output array! %d\n", x->cl_err);
        }
        
        struct timeval  tv1, tv2;
        gettimeofday(&tv1, NULL);
        
        clReleaseMemObject(cl_mem_input);
        clReleaseMemObject(cl_mem_output);
        
        gettimeofday(&tv2, NULL);
        
        post("Memory release time: %f seconds\n",
             (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
             (double) (tv2.tv_sec - tv1.tv_sec));
    }
}
