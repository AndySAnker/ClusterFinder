[ChemRxiv]  |  [Paper]

# ClusterFinder
Cluster is a novel, automated, high throughput screening approach which can find candidate structures for atomic pair distribution function (PDF) structural refinements. Finding starting models for PDF refinements is notoriously difficult when the PDF originates from nanoclusters or small nanoparticles. ClusterFinder algorithm is able to screen 10<sup>4</sup> – 10<sup>5</sup> candidate structures from structural databases such as the Inorganic Crystal Structure Database (ICSD) in minutes, using the crystal structures as templates in which it looks for atomic clusters that result in a PDF similar to the target measured PDF. The algorithm returns a rank-ordered list of clusters for further assessment by the user. ClusterFinder has performed well for simulated and measured PDFs of metal oxido clusters such as Keggin clusters. The approach is therefore a powerful approach to finding structural cluster candidates in a modelling campaign for PDFs of nanoparticles and nanoclusters.

1. [Installation](#installation)
2. [Usage on a single starting template](#usage-on-a-single-starting-template)
3. [Usage on a database of starting templates](#usageon-a-database-of-starting-templates)
4. [Authors](#authors)
5. [Cite](#cite)
6. [Contributing to the software](#contributing-to-the-software)
    1. [Reporting issues](#reporting-issues)
    2. [Seeking support](#seeking-support)

# Installation

To run ClusterFinder you will need some packages that can be installed through the requirement files.
```
pip install -r requirements.txt
``` 
You will also need to install [DiffPy-CMI](https://www.diffpy.org/products/diffpycmi/index.html) (see how to [HERE](https://www.diffpy.org/products/diffpycmi/index.html))

# Usage on a single starting template

See ClusterFinder.ipynb for an example of extracting a cluster from a starting template.

The [Experimental_Data](https://github.com/AndySAnker/ClusterFinder/tree/main/Experimental_Data) folder and the [Structure_Model](https://github.com/AndySAnker/ClusterFinder/tree/main/Structure_Models) folder provides examples of experimental data and starting templates.

# Usage on a database of starting templates

See ClusterFinder-databasescreen.ipynb for an example of extracting clusters from a database of starting template.

The [Experimental_Data](https://github.com/AndySAnker/ClusterFinder/tree/main/Experimental_Data) folder and the [Structure_Model](https://github.com/AndySAnker/ClusterFinder/tree/main/Structure_Models) folder provides examples of experimental data and starting templates. However, we are not allowed to distribute the entire database of CIFs.

# Authors
__Andy S. Anker__<sup>1</sup>   
 
<sup>1</sup> Department of Chemistry & Nano-Science Center, University of Copenhagen, Denmark

Should there be any questions, desired improvements or bugs please contact us on GitHub or 
through email: __andy@chem.ku.dk__.

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!

```
@article{anker2023clusterfinder,
title={ClusterFinder: A fast tool to find cluster structures from pair distribution function data},
author={Andy S. Anker, Ulrik Friis-Jensen, Frederik L. Johansen, Simon J. L. Billinge and Kirsten M. Ø. Jensen},
journal={ChemRxiv}
year={2023}}
```

# Contributing to the software

We welcome contributions to our software! To contribute, please follow these steps:

1. Fork the repository.
2. Make your changes in a new branch.
3. Submit a pull request.

We'll review your changes and merge them if they meet our quality standards.

## Reporting issues

If you encounter any issues or problems with our software, please report them by opening an issue on our GitHub repository. Please include as much detail as possible, including steps to reproduce the issue and any error messages you received.

## Seeking support

If you need help using our software, please reach out to us on our GitHub repository. We'll do our best to assist you and answer any questions you have.

