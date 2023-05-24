# Analysis of Subtractions including Stellar Proper Motion

Nearby stars will move on the scale of the LSST Survey.  This will create blurred PSFs in templates built from multilpe years and dipoles in subtractions across different years.  How significant are these effects, and does the LSST Science Pipelines code deal with them in a reasonable way?

This directory contains tutorial notebooks related to the exploration of the effects of proper motion on detected stars and building template.

All are welcome to contribute notebooks to this directory.


| Title | Description | Author |
|---|---|---|
| dia_proper_motion | Simulate a star that moves on the scale of an arcsecond during the 10 years of the survey.  Uses parts of the `source_injection` package to generate the synthetic stars.  Writes injected images into a new collection, and then does calexp-calexp subtractions from that new collection.  It then runs detection and measurement on the resulting subtractions and confirms that the DIA sources from the injected star are detected and well-characterized as a dipole.  This is an intermediate-level Notebook that calls Tasks on in-memory object and directly writes to a Butler. | Michael Wood-Vasey |
