This folder contains notebooks prepared by CSU students.

## Guide to Data Processing with Dense Basis Using an Automated Program (DB_*.ipynb)
AUTHOR: Aayush Joshi with Denvir Higgins and Giselle Martinez
Last verified: April 2024
Contact author: Aayush Joshi and Louise Edwards
Targeted learning level: some experience, undergraduate<br>

The objective of this notebook series is to employ an automated program to analyze the Truth Summary and Object Catalog from DP0 DC2. The primary focus is to query proximate, luminous galaxy clusters and ascertain their star formation history.

The link below provides documentation for Dense Basis installation, dependencies and other features. Proceed with this notebook once the installation is completed:
https://dense-basis.readthedocs.io/en/latest/
It will also be necessary to download LSST_list.dat

This notebook incorporates elements from tutorial notebooks 1-8 and is structured intachieve three main purpose:

1. **Program Utilization: DB_Guidebook** This section provides a comprehensive guide on how to effectively use the automated program. It includes step-by-step instructions and best practices for optimal results.
2. **Data Processing: DB_Processor** Here, you will find detailed procedures on how to process data from the Truth Summary and Object Catalog. This includes data cleaning, transformation, and preparation steps to ensure the data is ready for analysis.
3. **Data Visualization: DB_Visualizer** The final section focuses on visualizing the processed data. It demonstrates how to generate meaningful plots that can aid in understanding the star formation history of galaxy clusters.

By following this notebook, users will gain a thorough understanding of how to study galaxy clusters using the DP0 DC2 dataset, from initial data processing to final visualization.

## Mpl_vs_Bokey.ipynb

Author: Vicente Puga<br>
Contact Author: Louise Edwards ledwar04@calpoly.edu<br>
Last verified to run: 2023-04-02<br>
Targeted learning level: beginner, undergraduate<br>
Credit: This notebook uses code from DP0 Tutorials 1 and 2<br>

Description:

1. We extract Galaxies from the object catalogues using the TAP  

2. We explore creating color-magnitude diagrams for galaxies, with Matplotlib and Bokeh in order understand different functionalities 

2. We fit a line to the galaxy main sequence 

3. We add annotations to the plots in section 2.1 


## CSU_Higgins_BCHLensing.ipynb

Author: Denvir Higgins<br>
Contact Author: Louise Edwards ledwar04@calpoly.edu<br>
Last verified to run: 2023-06-02<br>
Targeted learning level: beginner<br>
Credit: The notebook utilizes aspects from tutorial notebooks 1-8<br>

Description: 
The purpose of the notebook is to use the Truth Summary catalog from DP0 DC2 to query nearby, bright galaxy clusters.
Within the notebook, you'll see how to porgram a simple equation, as well as make color images, and overplot a circle on an image.
You will identify brightest cluster galaxies (BCGs) and then use the distance and estimated total mass of the galaxy to determine the Einstein radius.
