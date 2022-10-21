# Well-sliced-metadynamics-reweighting
This code implements the reweighting scheme for well-sliced metadynamics (WSMD) reweighting using Python. Well-sliced metadynamics is an enchanced sampling method which uses a combination of umbrella sampling and well-tempered metadynamics in order to compute free energy surfaces with respect to two collective variables. In WSMD, umbrella sampling is run along one collective variable, while independent metadynamics biases are built up along a second collective variable. This code requires as input from those simulations, the location of umbrella biases, the force constants used in those biases, the values of the two collective variables sampling during the simulations, and [c(t) reweighting factors](https://www.plumed.org/doc-v2.7/user-doc/html/_m_e_t_a_d.html) computed by Plumed.

For more information, [read the attached PDF summarizing the equations used in the WSMD method](Well_Sliced_MetaDynamics_Equations.pdf). Additional details on the equations governing umbrella sampling and well-tempered metadynamics.

Code based on method developed by the Nair Group at IIT: Kanpur. For more information, look at the orginial publication:
Awasthi, S.; Kapil, V.; Nair, N. N. Sampling Free Energy Surfaces as Slices by Combining Umbrella Sampling and Metadynamics. J. Comput. Chem. 2016, 37 (16), 1413–1424. https://doi.org/10.1002/jcc.24349.
