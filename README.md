# sparsequanv
a repo storing code for only the sparse quanvolutional preprocessing layer for 3d sas data cubes

#### repo organization

this repo is organized into the quantum and classical counterparts.

the classical counterpart is fairly straightforward so we glance over it.

the quantum counterpart has two "philosophies" on approaching ML training:

1. Either as a preprocessing operation
2. Or as a quantum convolutional layer

We exploit the on-need basis computation of the quanvolutional layer to reach good efficiency when simulating. In practice, this wouldn't necessarily be the case. 

One may set "AS_PREPROCESSING = True" to switch between the two "philosophies" of training defined above. 