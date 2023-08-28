# Command Line Custom Coadd Optional Exercises Solutions

Contact Author: Aaron Meisner

Last verified: 2023 Aug 27

Targeted learning level: Advanced

Container size: large

## Preliminaries

The solutions to this command line tutorial, like the corresponding command line tutorial itself, need to be run using LSST pipelines version `w_2022_40`. For further elaboration on this important point [see the relevant DP0.2 command line tutorial](https://dp0-2.lsst.io/tutorials-examples/cmdline-custom-coadd.html).

Note also that the very first optional exercise from the DP0.2 command line custom coadd tutorial is a Jupyter notebook optional exercise rather than a command line optional exercise, and therefore no solution for that optional exercise is provided here.

You also must have a JupyterHub terminal window open and have run `setup lsst_distrib` to set up the LSST pipelines within your environment. Again, refer to [the relevant DP0.2 command line tutorial](https://dp0-2.lsst.io/tutorials-examples/cmdline-custom-coadd.html) for further setup details.

## First command line optional exercise

**Problem statement**: *Try applying further downstream processing to your custom coadd from the command line e.g., source detection run on the custom ``deepCoadd`` products.*

**Solution**:

Let's start out by looking at the pipeline definition for `step3`, which is the chunk of the overall pipeline from which you subselected the `makeWarp` and `assembleCoadd` tasks to run in the DP0.2 command line custom coadd tutorial. To do this, run `pipetask build` using the relevant pipeline definition YAML file (same one used throughout the command line custom coadd tutorial) and additionally specifying `#step3` in the URI:

```
pipetask build -p $DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml#step3 --show pipeline
```

Part of the output you will see as a result of this `pipetask build` command shows the list of `step3` Tasks:

```
  step3:
    subset:
    - forcedPhotCoadd
    - makeWarp
    - writeObjectTable
    - healSparsePropertyMaps
    - mergeMeasurements
    - selectGoodSeeingVisits
    - transformObjectTable
    - detection
    - consolidateObjectTable
    - deblend
    - templateGen
    - measure
    - assembleCoadd
    - mergeDetections
```

Note that the order in which these `step3` tasks are printed out may vary from one `pipetask build` call to the next, so do not worry if you see them listed in a different order.

Here you can see that there is a Task named `detection` which, just like `makeWarp` and `assembleCoadd`, is part of the `step3` pipeline chunk. So you can run source detection exactly as you did for `makewarp` and `assembleCoadd`, but now specifying `#detection` rather than `#makeWarp,assembleCoadd` in the pipeline URI. You also want to make sure to use your custom coadd for the source detection. This is accomplished by specifying the `-i` input to be the location you previously specified as the custom coadd output location in the command line custom coadd tutorial. Putting this together, one arrives at the following command:

```
LOGFILE=$LOGDIR/detection.log; \
date | tee $LOGFILE; \
pipetask --long-log --log-file $LOGFILE run --register-dataset-types \
-b dp02 \
-i u/$USER/custom_coadd_window1_cl00 \
-o u/$USER/custom_coadd_window1_cl00 \
-p $DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml#detection \
-d "tract = 4431 AND patch = 17 AND visit in (919515,924057,924085,924086,929477,930353) AND skymap = 'DC2'"; \
date | tee -a $LOGFILE
```

If you have not made a directory for log files and defined the associated `LOGFILE` environment variable, then you'll need to do that before running the above command. For instance, you could do:

```
mkdir logs
export LOGDIR=logs
```

Note that the above `pipetask run` command specifies the same output `-o` location as you used in the command line custom coadd tutorial. You could specify a different output location for the source detection outputs, if desired.

An example log file obtained by running the above `pipetask run` source detection command is provided here in the `logs` subdirectory, named `logs/detection.log`.

Running the above source detection command takes roughly a minute or slightly longer.

## Second command line optional exercise

**Problem statement**: *Try modifying another configuration parameter for the ``makeWarp`` and/or ``assembleCoadd`` tasks via the ``pipetask`` ``-c`` argument syntax.*

**Solution**:

First it's necessary to figure out what some other relevant parameters are. To do that, execute a `pipetask run` command with the ``--show config`` option specified:

```
pipetask run \
-b dp02 \
-p $DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml#makeWarp,assembleCoadd \
--show config
```

This `pipetask run` command again uses the same pipeline definition YAML file as did the DP0.2 command line custom coadd tutorial itself. The above command generates more than 1,000 lines of printouts total. Let's focus on the last few lines:

```
# Maximum variance scaling value to permit
config.scaleWarpVariance.limit=10.0

# Rescue artifacts from clipping that completely lie within a footprint detectedon the PsfMatched Template Coadd. Replicates a behavior of SafeClip.
config.doPreserveContainedBySource=True

# Ignore artifact candidates that are mostly covered by the bad pixel mask, because they will be excluded anyway. This prevents them from contributing to the outlier epoch count image and potentially being labeled as persistant.'Mostly' is defined by the config 'prefilterArtifactsRatio'.
config.doPrefilterArtifacts=True

# Prefilter artifact candidates that are mostly covered by these bad mask planes.
config.prefilterArtifactsMaskPlanes=['NO_DATA', 'BAD', 'SAT', 'SUSPECT']

# Prefilter artifact candidates with less than this fraction overlapping good pixels
config.prefilterArtifactsRatio=0.05

# Filter artifact candidates based on morphological criteria, i.g. those that appear to be streaks.
config.doFilterMorphological=False

# Grow streak footprints by this number multiplied by the PSF width
config.growStreakFp=5.0

No quantum graph generated or pipeline executed. The --show option was given and all options were processed.

```

For the sake of this optional exercise, let's somewhat arbitrarily choose to switch the `doFilterMorphological` parameter, which has to do with filtering artifacts like satellite streaks based on morphology, to `True` (the opposite of its default value). The DP0.2 simulated images do not have satellite streaks injected, but presumably such streaks will appear in real Rubin/LSST data.

Let's narrow things down by showing only the configuration for the `doFilterMorphological` parameter that will be changed:

```
pipetask run \
-b dp02 \
-p $DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml#makeWarp,assembleCoadd \
--show config=assembleCoadd::doFilterMorphological
```

The above command results in the following output:

```
Matching "doFilterMorphological" without regard to case (append :NOIGNORECASE to prevent this)
### Configuration for task `assembleCoadd'
# Filter artifact candidates based on morphological criteria, i.g. those that appear to be streaks.
config.doFilterMorphological=False
```

Which indeed shows only the configuration for `doFilterMorphological`, and also reaffirms that the default value of this parameter is `False`.

Now let's check that we can use the `-c` syntax of `pipetask run` to switch `doFilterMorphological` from its default value to `True` using the following command:

```
pipetask run \
-b dp02 \
-c assembleCoadd:doFilterMorphological=True \
-p $DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml#makeWarp,assembleCoadd \
--show config=assembleCoadd::doFilterMorphological
```

Note the addition of the `-c assembleCoadd:doFilterMorphological=True` option here. The `doFilterMorphological=True` assignment is prefaced with `assembleCoadd:` because `doFilterMorphological` is a parameter of the `assembleCoadd` Task.

```
Matching "doFilterMorphological" without regard to case (append :NOIGNORECASE to prevent this)
### Configuration for task `assembleCoadd'
# Filter artifact candidates based on morphological criteria, i.g. those that appear to be streaks.
config.doFilterMorphological=True

No quantum graph generated or pipeline executed. The --show option was given and all options were processed.
```


No quantum graph generated or pipeline executed. The --show option was given and all options were processed.

## Third command line optional exercise

**Problem statement**: *Try using the same two configuration parameter modifications as did this command line custom coadd tutorial, but implementing them via a separate configuration (``.py``) file, rather than via the ``pipetask`` ``-c`` argument (hint: to do this, you'd use the ``-C`` argument for ``pipetask``).*

**Solution**:

Within this solutions directory, there is a subdirectory called `config` with a file named `makeWarp_config.py`. The contents of this `config/makeWarp_config.py` file are as follows:

```
config.doApplyFinalizedPsf=False
config.connections.visitSummary="visitSummary"
```

Note that this is a Python file (`.py` extension), hence Python syntax is being used for e.g., boolean and string variables. The `makeWarp_config.py` file name is arbitrary -- this file could have been given some different name. Then in the `pipetask run` command that generates your custom coadd, simply delete the two lines that begin with `-c` and then add in a new line that begins with `-C` and specifies the relevant Task and Python config file:

```
LOGFILE=$LOGDIR/makeWarpAssembleCoadd-configfile-logfile.log; \
date | tee $LOGFILE; \
pipetask --long-log --log-file $LOGFILE run --register-dataset-types \
-b dp02 \
-i 2.2i/runs/DP0.2 \
-o u/$USER/custom_coadd_window1_cl00 \
-p $DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml#makeWarp,assembleCoadd \
-C makeWarp:config/makeWarp_config.py \
-d "tract = 4431 AND patch = 17 AND visit in (919515,924057,924085,924086,929477,930353) AND skymap = 'DC2'"; \
date | tee -a $LOGFILE
```

If you have not made a directory for log files and defined the associated `LOGFILE` environment variable, then you'll need to do that before running the above command. For instance, you could do:

```
mkdir logs
export LOGDIR=logs
```


In the newly added `pipetask run` line starting with `-C`, the name of the separate config file is prefaced with `makeWarp:` because the parameters specified in `makeWarp_config.py` pertain to the `makeWarp` Task (as opposed to e.g., the `assembleCoadd` Task).

An example log file obtained by running the above `pipetask run` command with `-C` is provided here in the `logs` subdirectory, named `logs/makeWarpAssembleCoadd-configfile-logfile.log`.

The `pipetask run` command from this optional exercise solution takes ~30-35 minutes to run.

## Fourth command line optional exercise

**Problem statement**: *Run the ``pipetask qgraph`` command from section 3.1, but with the final line ``--show graph`` removed. This still takes roughly 15 minutes, but prints out a much more concise summary listing only the total number of quanta to be executed, which should be 7.*

**Solution**:

Execute the following command:

```
pipetask qgraph \
-b dp02 \
-i 2.2i/runs/DP0.2 \
-p $DRP_PIPE_DIR/pipelines/LSSTCam-imSim/DRP-test-med-1.yaml#makeWarp,assembleCoadd \
-c makeWarp:doApplyFinalizedPsf=False \
-c makeWarp:connections.visitSummary="visitSummary" \
-d "tract = 4431 AND patch = 17 AND visit in (919515,924057,924085,924086,929477,930353) AND skymap = 'DC2'"
```

This will result in the following much-abbreviated printout, which states that there are 7 quanta in the QuantumGraph, consistent with what you saw in the more verbose output:

```
lsst.ctrl.mpexec.cmdLineFwk INFO: QuantumGraph contains 7 quanta for 2 tasks, graph ID: '1693040624.3335783-1840'
```

The `pipetask qgraph` command from this optional exercise solution takes ~15 minutes to run.
