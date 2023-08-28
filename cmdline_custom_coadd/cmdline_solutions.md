# Command Line Custom Coadd Optional Exercises Solutions

Contact Author: Aaron Meisner

Last verified: 2023 Aug 27

Targeted learning level: Advanced

Container size: large

## Warning

The solutions to this command line tutorial, like the corresponding command line tutorial itself, need to be run using LSST pipelines version `w_2022_40`. For further elaboration on this important point, [see the relevant DP0.2 command line tutorial](https://dp0-2.lsst.io/tutorials-examples/cmdline-custom-coadd.html).

Note also that the very first optional exercise from the DP0.2 command line custom coadd tutorial is a Jupyter notebook optional exercise rather than a command line optional exercise, and therefore no solution for that optional exercise is provided here.

## First command line optional exercise

**Problem statement**: *Try applying further downstream processing steps to your custom coadd from the command line e.g., source detection run on the custom ``deepCoadd`` products.*

**Solution**:

## Second command line optional exercise

**Problem statement**: *Try modifying other configuration parameters for the ``makeWarp`` and/or ``assembleCoadd`` tasks via the ``pipetask`` ``-c`` argument syntax.*

**Solution**:

## Third command line optional exercise

**Problem statement**: *Try using the same two configuration parameter modifications as did this tutorial, but implementing them via a separate configuration (``.py``) file, rather than via the ``pipetask`` ``-c`` argument (hint: to do this, you'd use the ``-C`` argument for ``pipetask``).*

**Solution**:

## Fourth command line optional exercise

**Problem statement**: *Run the ``pipetask qgraph`` command from section 3.1, but with the final line ``--show graph`` removed. This still takes roughly 15 minutes, but prints out a much more concise summary listing only the total number of quanta to be executed, which should be 7.*

**Solution**:

