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


void *graf_new(t_symbol *s, long argc, t_atom *argv)
{
    t_graf *x = (t_graf *)object_alloc(graf_class);
    
    if (x) {
        post("compilation working");
        dsp_setup((t_pxobject *)x, 2);
        outlet_new(x, "signal");
        // disable in-place optimisation so that the input and output buffers are distinct
        x->ob.z_misc |= Z_NO_INPLACE;
        
        // initialize stuff
    }
    return (x);
}

void graf_free(t_graf *x)
{
    dsp_free((t_pxobject *)x);
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
    t_double *out = outs[0];    // we get audio for each outlet of the object from the **outs argument
    
    // clear output buffer
    for(long n = 0; n < sampleframes; n++) {
        out[n] = 0;
    }
}
