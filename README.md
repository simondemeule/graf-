# graf-
Experimental GPU-based audio processor for Max MSP made with OpenCL

# Why does this thing exist?
This thing exist because I was curious to know whether it was possible to run some parallel algorithms at audio rate on the GPU. The short answer is yes. The long answer is that it isn't very practical: large buffer sizes are necessary to counteract delays created by system overhead and limitations imposed by memory bandwidth, and the GPU only really starts outperforming the CPU when buffer sizes go well past practical sizes. This was tested using a 2019 15" MacBook Pro, a machine that has a better than average processor and graphics card, so it likely is the case that this isn't practical on most computers, at least in 2020.

# What does it even do?
This is a very stripped down, basic implementation of a 2D IRR filter. This filter doesn't operate on raw signal samples: it operates on the FFT transform of the signal. This allows to create really weird, funky spectral filters. Mostly, this is just a program containing the basic bits of OpenCL code necessary to get the GPU crunching some numbers. The filter kernel it uses is a bit broken, and it lacks stereo I/O necessary for processing the sine and cosine components of the transform properly.

# How do I use it?
You can open the patch in /grafproj/ to play around with the compiled external, properly setup in a pfft~, so that the FFT is applied properly on the signal. 
If you want to develop your own Max external using OpenCL for computing things, feel free to use this as a starting point!

# An extra note
OpenCL can use either CPUs or GPUs for crunching the numbers. Somehow, the program needs to decide which device to use. Because this was written in a hurry, selection of the device is hard-coded. Go mess around with line 155 in graf~.c if the current hard-coded device selection doesn't pick the compute device you need.
