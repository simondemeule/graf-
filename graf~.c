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

// struct to represent the object's state
typedef struct _graf {
    t_pxobject ob;            // the object itself (t_pxobject in MSP instead of t_object)
    double samplerate;
    long sampleframes;
    t_double *in;
    t_double *out;
    
    bool cl_pending = false;
    bool cl_init_attempted = false;
    bool cl_init_failed = false;
    size_t cl_global;                      // global domain size for our calculation
    size_t cl_local;                       // local domain size for our calculation
    cl_int cl_err;
    int cl_device_idx;
    cl_device_id cl_device_id;             // compute device id
    cl_context cl_context;                 // compute context
    cl_command_queue cl_commands;          // compute command queue
    cl_program cl_program;                 // compute program
    cl_kernel cl_kernel;                   // compute kernel
    cl_mem cl_mem_input;
    cl_mem cl_mem_output;
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

const char *kernel_source = "\n" \
"__kernel void graf(                                                    \n" \
"   __global double* input,                                             \n" \
"   __global double* output,                                            \n" \
"   const long count)                                                   \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count) {                                                     \n" \
"       output[i] = input[i];                                           \n" \
"   }                                                                   \n" \
"}                                                                      \n" \
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
        
        post(" %s\n", device_name);
    }
    
    // hardcoded device selection
    
#define D_VEGA20 (2)
#define D_UHD630 (1)
#define D_I9 (0)
    
    x->cl_device_id = device_ids[D_UHD630];
    
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
    
    post("Device chosen for computation:\n %s\n", device_name);
    
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
    
    // Create a command commands
    //
    x->cl_commands = clCreateCommandQueue(x->cl_context, x->cl_device_id, 0, &(x->cl_err));
    if (!x->cl_commands)
    {
        post("Error: Failed to create a command commands!\n");
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
    x->cl_kernel = clCreateKernel(x->cl_program, "graf", &(x->cl_err));
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
    
    x->cl_mem_input = clCreateBuffer(x->cl_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * x->sampleframes, x->in, &(x->cl_err));
    if (x->cl_err != CL_SUCCESS)
    {
        post("Error: Failed to attach input memory! %d\n", x->cl_err);
        x->cl_init_failed = true;
        return;
    }
    x->cl_mem_output = clCreateBuffer(x->cl_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * x->sampleframes, x->out, &(x->cl_err));
    if (x->cl_err != CL_SUCCESS)
    {
        post("Error: Failed to attach output memory! %d\n", x->cl_err);
        x->cl_init_failed = true;
        return;
    }
    
    // Set the arguments to our compute kernel
    //
    x->cl_err  = 0;
    x->cl_err  = clSetKernelArg(x->cl_kernel, 0, sizeof(cl_mem), &(x->cl_mem_input));
    x->cl_err |= clSetKernelArg(x->cl_kernel, 1, sizeof(cl_mem), &(x->cl_mem_output));
    x->cl_err |= clSetKernelArg(x->cl_kernel, 2, sizeof(long), &(x->sampleframes));
    if (x->cl_err != CL_SUCCESS)
    {
        post("Error: Failed to set kernel arguments! %d\n", x->cl_err);
        x->cl_init_failed = true;
        return;
    }
    
    x->cl_init_failed = false;
}

void graf_cl_reset(t_graf *x) {
    clReleaseMemObject(x->cl_mem_input);
    clReleaseMemObject(x->cl_mem_output);
    clReleaseProgram(x->cl_program);
    clReleaseKernel(x->cl_kernel);
    clReleaseCommandQueue(x->cl_commands);
    clReleaseContext(x->cl_context);
}

void *graf_new(t_symbol *s, long argc, t_atom *argv)
{
    t_graf *x = (t_graf *)object_alloc(graf_class);
    
    if (x) {
        dsp_setup((t_pxobject *)x, 1);
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
    
    // instead of calling dsp_add(), we send the "dsp_add64" message to the object representing the dsp chain
    // the arguments passed are:
    // 1: the dsp64 object passed-in by the calling function
    // 2: the symbol of the "dsp_add64" message we are sending
    // 3: a pointer to your object
    // 4: a pointer to your 64-bit perform method
    // 5: flags to alter how the signal chain handles your object -- just pass 0
    // 6: a generic pointer that you can use to pass any additional data to your perform method
    
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
    // possible edge case: change of sampleframes? is the graph rebuilt then?
    
    if(!x->cl_init_attempted) {
        x->in = ins[0];
        x->out = outs[0];
        x->sampleframes = sampleframes;
        graf_cl_init(x);
        x->cl_init_attempted = true;
    }
    if(!x->cl_init_failed)
    {
        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //
        x->cl_global = sampleframes;
        if(x->cl_pending) {
            post("Can't keep up! Last command is still incomplete!");
        }
        x->cl_pending = true;
        x->cl_err = clEnqueueNDRangeKernel(x->cl_commands, x->cl_kernel, 1, NULL, &(x->cl_global), &(x->cl_local), 0, NULL, NULL);
        if (x->cl_err)
        {
            post("Error: Failed to execute kernel!\n");
        }
        
        // Wait for the command commands to get serviced before reading back results
        //
        clFinish(x->cl_commands);
        x->cl_pending = false;
    }
}
