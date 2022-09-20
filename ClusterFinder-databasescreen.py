import numpy as np
import multiprocessing as mp
from operator import itemgetter
import time, glob, pdb, tqdm, os, shutil
import matplotlib as mpl
import matplotlib.cm
from itertools import repeat
from diffpy.Structure import Structure, Atom
from diffpy.srfit.pdf import PDFContribution, PDFParser, PDFGenerator
from diffpy.srfit.fitbase import FitRecipe, FitResults, Profile, FitContribution
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from scipy.optimize.minpack import leastsq

def structure_catalogue_maker(Number_of_atoms):
    """Makes a catalogue of structures"""
    
    structure_catalogue = np.ones((Number_of_atoms,Number_of_atoms))
    structure_catalogue[np.array([range(Number_of_atoms)]),np.array([range(Number_of_atoms)])] = 0
    return structure_catalogue

def Load_startmodel(starting_model):
    # Read structure and divide it into two lists: Atoms we want to iterate (W) and atoms we do not iterate (O)
    struct=[]
    with open(starting_model, 'r') as fi:
        for line in fi.readlines():
            sep_line=line.strip('{}\n\r ').split()
            if len(sep_line)==4: #  tillader andre informationer i xyz filen some ikke skal laeses
                struct.append(sep_line)
    elements=np.array(struct)[:,0]
    xyz=(np.array(struct)[:,1:].astype(float))
    
    return elements, xyz

def fitting(structure_catalogue, Experimental_Data, xyz, atom_ph, Qmin, Qmax, Qdamp, rmin, rmax, index):
    """This function takes in a 'starting_model', and an 'index' from the 'structure_catalogue'. It generates the 
    corresponding structure and fit it to the 'Experimental_Data'."""
    NumMo = np.shape(structure_catalogue)[1]
    xyz_Mo = xyz[:NumMo].copy()
    xyz_O = xyz[NumMo:len(xyz)].copy()
    keep_O = np.zeros(len(xyz_O))
    # Cycle through W atoms and delete W according to index 0's from permutation
    delete_M = np.where(np.array(structure_catalogue)[index,:] == 0)[0]

    # Delete atoms from starting model 
    xyz_Mo = np.delete(xyz_Mo, delete_M, 0)

    # Cycle through all atoms that is not iteratable and test if it is within the threshold distance. Delete atoms with no bonds
    for j in range(len(xyz_O)):
        dists = np.sqrt((xyz_Mo[:,0]-xyz_O[j,0])**2+(xyz_Mo[:,1]-xyz_O[j,1])**2+(xyz_Mo[:,2]-xyz_O[j,2])**2)
        if np.min(dists) < threshold:    
            keep_O[j] = 1

    # Cycle through W atoms and delete W according to index 0's from permutation
    delete_O = np.where(np.array(keep_O) == 0)[0]
    # Delete atoms from starting model 
    xyz_O = np.delete(xyz_O, delete_O, 0)

    # Create structure for iterable (W) and non-iterable (O) atoms and combine them
    Mo_cluster = Structure([Atom(atom_ph, xi) for xi in xyz_Mo])
    O_cluster = Structure([Atom('O', xi) for xi in xyz_O])
    cluster = Mo_cluster + O_cluster
    
    # Make a standard cluster refinement using Diffpy-CMI
    # Import the data and make it a PDFprofile. Define the range of the data that will be used in the fit.
    pdfprofile = Profile()
    pdfparser = PDFParser()
    pdfparser.parseFile(Experimental_Data)
    pdfprofile.loadParsedData(pdfparser)
    pdfprofile.setCalculationRange(xmin = rmin, xmax = rmax)

    # Setup the PDFgenerator that calculates the PDF from the structure
    pdfgenerator_cluster = PDFGenerator("G")
    # Add the profile and both generators to the PDFcontribution
    pdfcontribution = FitContribution("pdf")
    pdfcontribution.setProfile(pdfprofile, xname="r") 
    pdfcontribution.addProfileGenerator(pdfgenerator_cluster)
    
    pdfgenerator_cluster.setQmin(Qmin)
    pdfgenerator_cluster.setQmax(Qmax)
    pdfgenerator_cluster._calc.evaluatortype = 'OPTIMIZED'
    pdfgenerator_cluster.setStructure(cluster, periodic = False)

    # Use scaling factors proportional to molar content
    pdfcontribution.setEquation('mc*G')

    # Define the recipe to do the fit and add it to the PDFcontribution
    recipe = FitRecipe()
    recipe.addContribution(pdfcontribution)

    # Avoid too much output during fitting 
    recipe.clearFitHooks()

    # Add the scale factor.
    recipe.addVar(pdfcontribution.mc, 1.0, tag = "scale")
    
    # Add the instrumental parameters to the two generators
    pdfgenerator_cluster.qdamp.value = Qdamp
    
    # Add ADP and "cell" for the cluster
    phase_cluster = pdfgenerator_cluster.phase
    atoms = phase_cluster.getScatterers()
    lat = phase_cluster.getLattice()

    recipe.newVar("zoomscale", 1.0, tag = "lat")
    recipe.constrain(lat.a, 'zoomscale')
    recipe.constrain(lat.b, 'zoomscale')
    recipe.constrain(lat.c, 'zoomscale')
    recipe.restrain("zoomscale", lb=0.95, ub = 1.05, sig=0.001)
    
    Mo_cluster = recipe.newVar("Mo_Biso_cluster", 0.3, tag = 'adp_Mo')
    O_cluster = recipe.newVar("O_Biso_cluster", 0.4, tag = 'adp_O')

    for atom in atoms:
        if atom.element.title() == atom_ph:
            recipe.constrain(atom.Biso, Mo_cluster)
        elif atom.element.title() == "O":
            recipe.constrain(atom.Biso, O_cluster)
  
    #free parameters are set
    recipe.fix('all')
    recipe.free("scale")
    leastsq(recipe.residual, recipe.getValues())
    #recipe.free("lat")
    #leastsq(recipe.residual, recipe.getValues())
   
    # We calculate the goodness-of-fit, Rwp
    g = recipe.pdf.profile.y
    gcalc = recipe.pdf.evaluate()
    Rwp = np.sqrt(sum((g - gcalc)**2) / sum((g)**2))
    
    return Rwp

def MotEx(inputs):
    XYZ_file, Metal_Atoms, Experimental_Data, max_atoms_supercell, atom_ph, Qmin, Qmax, Qdamp, rmin, rmax, SaveResults = inputs
    print (" XYZ File: ", XYZ_file)
    start_time = time.time()
    # Load data and start model
    elements, xyz = Load_startmodel(XYZ_file)
    if len(xyz) < max_atoms_supercell:
        NumMo = 0
        for Metal_Atom in Metal_Atoms:
            NumMo += list(elements).count(Metal_Atom)
        if NumMo > 1:
            # Step 1: Make the structure catalogue
            structure_catalogue = structure_catalogue_maker(Number_of_atoms=NumMo)

            ### Step 2: Produce organized structure catalogue with Rwp values
            Result = []
            for i in range(len(structure_catalogue)):
                Rwp = fitting(structure_catalogue, Experimental_Data, xyz, atom_ph, Qmin, Qmax, Qdamp, rmin, rmax, i)
                Result.append(Rwp)
            Result = np.column_stack([Result, np.asarray(structure_catalogue)])
            
            # Step 3: Calculate Atom Contribution values
            m, AtomContributionValues = calculate_atomContributionValue(Result, SaveResults)
            Mean_AtomContributionValue = np.mean(AtomContributionValues)
            STD_AtomContributionValue = np.std(AtomContributionValues)

            # Step 4: Output a CrystalMaker file
            Make_CrystalMakerFile(NumMo, elements, xyz, AtomContributionValues, m, SaveResults+XYZ_file.replace(XYZ_file[:XYZ_file.find("icsd")], "")[:-4], threshold)
            Make_VestaFile(NumMo, elements, xyz, AtomContributionValues, m, SaveResults+XYZ_file.replace(XYZ_file[:XYZ_file.find("icsd")], "")[:-4], threshold)
            Rwp = Result[:,0].mean()
        else:
            print ("There is not any of the specified atoms - "+ str(Metal_Atoms) +" - in the CIF file - " + XYZ_file + "\n")
            Rwp = 10
            Mean_AtomContributionValue = 10
            STD_AtomContributionValue = 0
    else:
        print ("Structure is too big - has "+ str(len(xyz)) +" atoms and threshold is " + str(max_atoms_supercell) + "\n")
        Rwp = 10
        Mean_AtomContributionValue = 10
        STD_AtomContributionValue = 0


    return XYZ_file, Rwp, Mean_AtomContributionValue, STD_AtomContributionValue

def calculate_atomContributionValue(Result, saveResults):
    """Calculate atom contribution value list from the result array"""
    
    # Define AtomContributionValues vector
    AtomContributionValues = Result[:,0]
    
    # Normalise the AtomContributionValues
    AtomContributionValues_ph = AtomContributionValues.copy()
    AtomContributionValues_ph.sort()

    # Define colormap of viridis.reverse
    norm = mpl.colors.Normalize(vmin=AtomContributionValues_ph[round((len(AtomContributionValues))/10)], vmax=AtomContributionValues_ph[-round((len(AtomContributionValues))/10)])
    cmap = matplotlib.cm.cividis
    m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Save results to file
    f = open(saveResults+"AtomContributionValues_MotEx.txt", "w")
    f.write("\nAtom contribution are calculated to: \n")
    for i in range(len(AtomContributionValues)):
        f.write("Atom # "+ str(i+1) + ":  "+ str(AtomContributionValues[i]) + "  Colorcode:  "+ mpl.colors.rgb2hex(m.to_rgba(AtomContributionValues[i]))+"\n")
    
    return m, AtomContributionValues

def Make_CrystalMakerFile(NumMo, elements, xyz, AtomContributionValues, m, saveResults, threshold):
    # Read bonds and colors of all atoms
    bonding = []
    with open("Bonding.txt", 'r') as fi:
        for line in fi.readlines():
            sep_line=line.strip('{}\n\r ').split()
            bonding.append(sep_line)
    bonding = np.array(bonding)
    
    # Output a crystalmaker file to visualize the results
    CrystalMaker = open(saveResults+'_CrystalMaker.cmtx', 'w')

    CrystalMaker.write("MOLE  CrystalMaker molecule format\n")
    CrystalMaker.write("TITL  Molecule\n\n")
    CrystalMaker.write("! Model type\n")
    CrystalMaker.write("MODL  1\n\n")

    CrystalMaker.write("! Depth fading settings\n")
    CrystalMaker.write("DCUE  1.000000 0.212899 0.704686\n\n")

    CrystalMaker.write("! Colour definitions:\n")
    CrystalMaker.write("TYPE\n")

    # Assign colors to all the atoms
    for iter, element in enumerate(elements):
        if iter < NumMo:
            #CrystalMaker.write(element + str(iter+1) + " 1.32 ")
            CrystalMaker.write(element + str(iter+1) + " " + bonding[np.where(bonding == element)[0][0], 1] + " ")
            rgb1 = m.to_rgba(AtomContributionValues[iter])[:-1][0]
            rgb2 = m.to_rgba(AtomContributionValues[iter])[:-1][1]
            rgb3 = m.to_rgba(AtomContributionValues[iter])[:-1][2]
            CrystalMaker.write(str(rgb1) + " " + str(rgb2) + " " + str(rgb3))
            CrystalMaker.write("\n")
        else:
            #CrystalMaker.write(element + str(iter+1) + " 0.66 ")
            CrystalMaker.write(element + str(iter+1) + " " + bonding[np.where(bonding == element)[0][0], 1] + " ")
            rgb1 = int(float(bonding[np.where(bonding == element)[0][0], 2])*255) #mpl.colors.to_rgb("#FF0000")[0]
            rgb2 = int(float(bonding[np.where(bonding == element)[0][0], 3])*255) #mpl.colors.to_rgb("#FF0000")[1]
            rgb3 = int(float(bonding[np.where(bonding == element)[0][0], 4])*255) #mpl.colors.to_rgb("#FF0000")[2]
            CrystalMaker.write(str(rgb1) + " " + str(rgb2) + " " + str(rgb3))
            CrystalMaker.write("\n")
    
    CrystalMaker.write("\n")
    CrystalMaker.write("! Atoms list\n")
    CrystalMaker.write("! Bond Specifications\n")
    
    # Assign bonds between the atoms
    for iter, element in enumerate(elements):
        if iter < NumMo:
            NI_elements = np.delete(np.unique(elements), np.where(np.unique(elements) == element)[0])
            for NI_element in NI_elements:
                CrystalMaker.write("BMAX " + element + " " + str(NI_element) + "  " + str(threshold))
                CrystalMaker.write("\n")
    
    CrystalMaker.write("\n")
    CrystalMaker.write("! Atoms list\n")
    CrystalMaker.write("ATOM\n")
    
    # Assign coordinates to the atoms
    for iter, element in enumerate(elements):
        if iter < NumMo:
            CrystalMaker.write(element + " " + element + str(iter+1) + " " + str(xyz[iter][0]) + " " + str(xyz[iter][1]) + " " + str(xyz[iter][2]) + "\n")
        else:
            CrystalMaker.write(element + " " + element + str(iter+1) + " " + str(xyz[iter][0]) + " " + str(xyz[iter][1]) + " " + str(xyz[iter][2]) + "\n")

    CrystalMaker.close()
    
    return None

def Make_VestaFile(NumMo, elements, xyz, AtomContributionValues, m, saveResults, threshold):
    # Read bonds and colors of all atoms
    bonding = []
    with open("Bonding.txt", 'r') as fi:
        for line in fi.readlines():
            sep_line=line.strip('{}\n\r ').split()
            bonding.append(sep_line)
    bonding = np.array(bonding)

    # Output a Vesta file to visualize the results
    Vesta = open(saveResults+'_Vesta.vesta', 'w')

    Vesta.write("#VESTA_FORMAT_VERSION 3.5.4\n\n\n")
    Vesta.write("MOLECULE\n\n")
    Vesta.write("Title\n")
    Vesta.write("XYZ file\n\n")

    Vesta.write("STRUC\n")
    # Assign coordinates to the atoms
    for iter, element in enumerate(elements):
        Vesta.write(str(iter+1) + " " + element + " " + element + str(iter+1) + " 1.0000 " + str(xyz[iter][0]) + " " + str(xyz[iter][1]) + " " + str(xyz[iter][2]) + "1" + " -" + "\n")
        Vesta.write("0 0 0 0\n")
    Vesta.write("  0 0 0 0 0 0 0\n")

    Vesta.write("SBOND\n")
    # Assign bonds between the atoms
    unique_elements = np.unique(elements)
    for iter, element1 in enumerate(unique_elements):
      for iter, element2 in enumerate(unique_elements):
        if not element1 == element2:
          Vesta.write(str(iter+1) + " " + element1 + " " + element2 + " 0.0 " + str(threshold) + " 0 1 1 0 1 0.25 2 127 127 127\n")
          Vesta.write("0 0 0 0\n")
    
    Vesta.write("SITET\n")
    # Assign colors to all the atoms
    for iter, element in enumerate(elements):
        if iter < NumMo:
            rgb1 = int(m.to_rgba(AtomContributionValues[iter])[:-1][0]*255)
            rgb2 = int(m.to_rgba(AtomContributionValues[iter])[:-1][1]*255)
            rgb3 = int(m.to_rgba(AtomContributionValues[iter])[:-1][2]*255)
            Vesta.write(str(iter+1) + " " + element + str(iter+1) + " " + bonding[np.where(bonding == element)[0][0], 1] + " " + str(rgb1) + " " + str(rgb2) + " " + str(rgb3) + " " + str(rgb1) + " " + str(rgb2) + " " + str(rgb3) + " 204 0\n")
        else:
            rgb1 = int(float(bonding[np.where(bonding == element)[0][0], 2])*255)
            rgb2 = int(float(bonding[np.where(bonding == element)[0][0], 3])*255)
            rgb3 = int(float(bonding[np.where(bonding == element)[0][0], 4])*255)
            Vesta.write(str(iter+1) + " " + element + str(iter+1) + " " + bonding[np.where(bonding == element)[0][0], 1] + " " + str(rgb1) + " " + str(rgb2) + " " + str(rgb3) + " " + str(rgb1) + " " + str(rgb2) + " " + str(rgb3) + " 204 0\n")
    Vesta.write("0 0 0 0 0 0\n")
    
    Vesta.write("ATOMT\n")
    done_deal_atoms = []
    for iter, element in enumerate(elements):
      if element not in done_deal_atoms:
        rgb1 = int(float(bonding[np.where(bonding == element)[0][0], 2])*255)
        rgb2 = int(float(bonding[np.where(bonding == element)[0][0], 3])*255)
        rgb3 = int(float(bonding[np.where(bonding == element)[0][0], 4])*255)
        Vesta.write(str(iter+1) + " " + element + " " + bonding[np.where(bonding == element)[0][0], 1] + " " + str(rgb1) + " " + str(rgb2) + " " + str(rgb3) + " " + str(rgb1) + " " + str(rgb2) + " " + str(rgb3) + " 204\n")
        done_deal_atoms.append(element)
    Vesta.write("0 0 0 0 0 0\n")

    Vesta.close()
    
    return None

def prepare_folders(SaveResults):
    # Check whether the specified path exists or not
    isExist = os.path.exists(SaveResults)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(SaveResults)

    isExist = os.path.exists(SaveResults[:-1]+"_top5")
    if not isExist:
      os.makedirs(SaveResults[:-1]+"_top5")

    return None

### First define the XYZ  path and the experimental dataFile name
XYZ_path = "2017p1XYZs/2017p1XYZs_cleaned/"
StemName = "DanMAX_AlphaKeggin_nyquist"
SaveResults = "Results_AlphaKeggin_opt/"
Metal_Atoms = ["Mo", "W", "Fe"]

### First define the experimental data path and the path you want the structure catalogue with fits to be saved
Experimental_Data = "Experimental_Data/"+StemName+".gr" # Name of the experimental file
XYZ_files = glob.glob(XYZ_path+"*.xyz")
threshold = 2.6 # Longest bond distance between the metal and oxygen atom
atom_ph, Qmin, Qmax, Qdamp, rmin, rmax = "W", 0.7, 20, 0.05, 1.6, 10
max_atoms_supercell = 1000 # Maximum number of allowed atoms in supercell. 500 is default. Larger value excludes less supercells but makes the algorithm slower.
cores = None
prepare_folders(SaveResults)

start_time = time.time()
inputs = zip(XYZ_files, repeat(Metal_Atoms), repeat(Experimental_Data), repeat(max_atoms_supercell), repeat(atom_ph), repeat(Qmin), repeat(Qmax), repeat(Qdamp), repeat(rmin), repeat(rmax), repeat(SaveResults))
Resultdict = {}
ResultBestdict = {}
# Distribute the job out on the entire computer
with mp.Pool(processes=cores) as pool:
    with tqdm.tqdm(total=len(XYZ_files)) as pbar:
        for XYZ_file, Rwp, Mean_AtomContributionValue, STD_AtomContributionValue in pool.imap_unordered(MotEx, inputs):
            if Mean_AtomContributionValue < 10:
                Resultdict.update({XYZ_file: Mean_AtomContributionValue})
                if len(ResultBestdict) < 5:
                    ResultBestdict.update({XYZ_file: Mean_AtomContributionValue})
                else:
                    if Rwp < max(ResultBestdict.values()):
                        ResultBestdict.pop(max(ResultBestdict, key=ResultBestdict.get), None)
                        ResultBestdict.update({XYZ_file: Mean_AtomContributionValue})
            pbar.update()

sort_orders = sorted(ResultBestdict.items(), key=itemgetter(1))
print ("Top 5: ", sort_orders[:5])
print ("Total time: ", time.time() - start_time, " s")

# Save top 5 in the Results_top5 folder
for i in range(5):
    shutil.copyfile(SaveResults + sort_orders[i][0][sort_orders[i][0].find("icsd"):-4] +'_CrystalMaker.cmtx', SaveResults[:-1] + "_top5/" + sort_orders[i][0][sort_orders[i][0].find("icsd"):-4] +'_CrystalMaker.cmtx')
    shutil.copyfile(SaveResults + sort_orders[i][0][sort_orders[i][0].find("icsd"):-4] +'_Vesta.vesta', SaveResults[:-1] + "_top5/" + sort_orders[i][0][sort_orders[i][0].find("icsd"):-4] +'_Vesta.vesta')
