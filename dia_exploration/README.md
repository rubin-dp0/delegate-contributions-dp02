# Analysis of DIA sources
DIA sources are detections of transient and variable sources on difference images. 

This directory contains tutorial notebooks related to the inspection, analysis, and simulation of variable sources.

All are welcome to contribute notebooks to this directory.


| Title | Description | Author |
|---|---|---|
| dia_supernova | Plot lightcurves of Supernovae detected on difference images, along with cutouts of template, calexp and diffexp images. Provide hints on how to produce difference images with user defined templates. | Vincenzo Petrecca |
| dia_match_truth | Generate a catalog of SNe Ia detected on the DP0.2 difference images, match it with a catalog of DP0.2 TruthSummary SNe Ia using the astropy function match_coordinates_sky, and analyze the results. | Douglas Tucker |
| dia_template_contamination | Demonstrates a way to evaluate and correct for template contamination (transient flux in the template images which leads to over-subtracted fluxes in the difference image). | Melissa Graham |
| dia_SNIa_host_association | Demonstrates and evaulates the efficacy of several methods for associating SNIa with their host galaxies, using galaxy shape parameters in the Object catalog.  | Melissa Graham |
| dia_truth_host_sample | Demonstrates how to assemble a sample of true SN host galaxies that were detected in the deepCoadds. | Melissa Graham |
| dia_dipoles_from_proper_motion | Simulate a star that moves on the scale of an arcsecond during the 10 years of the survey.  Uses parts of the `source_injection` package to generate the synthetic stars.  Writes injected images into a new collection, and then does calexp-calexp subtractions from that new collection.  It then runs detection and measurement on the resulting subtractions and confirms that the DIA sources from the injected star are detected and well-characterized as a dipole.  This is an intermediate-level Notebook that calls Tasks on in-memory object and directly writes to a Butler. | Michael Wood-Vasey |
