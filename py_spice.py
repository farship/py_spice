"""
py_spice.py
A program to calculate circuit values for a given frequency range.
The program takes in a .net file as the input that defines the circuit nodes, components and connections in the <CIRCUIT> section
The <TERMS> section defines the input voltage/current, the source resistance/conductance, the load impedance and the frequency range to test over, linearly spaced or log spaced
The <OUTPUT> section defines the output variables and their units.
The program also takes a file name for the output .csv file, which it creates or overwrites with the output variables and associated frequency.
"""

"""
cProfile allows the excecution time for each function and all called function,
as well as the number of times these are called. 
It is used for finding slow parts of the code, 
using a larger circuit and number of frequencies allows for more accurate profiling.
Uncomment the import and change main(argv) call to be profile.run('main(argv)'); located at the end of the code
"""
# import cProfile as profile # used for finding slow/unoptimised functions
from sys import argv # Take in the arguments from the command line, for the file names.
import numpy as np # Arrays and matrix calculations optimised with C and pre-compiled functions.
"""
The main function is used for calling all functions in the order required.
Try...Execept are used for error handling.
Print statements are used to indicate what steps have been started, indicating the slow sections and helping to debug runtime errors

"""
def main(args = []): # Defaults args to be an empty list and specifies the type.
    try: # Handles not enough arguments
        print("Starting Py_SPICE Program using:\nNet File = {}\nResults File = {}".format(args[1], args[2])) # Reports to the user the files that are being read/written to
    except Exception as err:
        print(err, ": Argument 1 must be the net file, Argument 2 must be the output file.")
        print("Arguments given ", args)
        exit() # Exit the program
        
    try: # Though 'w' should make the file and then continue running, python tends to make the file and error out
        open(args[2], 'x').close() # if file doesn't exist it will make it, then close to avoid overwriting later
        print("Creating output file\"{file}\"".format(file=args[2]))
    except:
        open(args[2], 'w').close() # if it does exist it will erase the contents, then close to avoid overwriting later
        print("Erasing output file contents...")
    
    net = importNet(args[1]) # returns the contents of the net file

    strippedNet = stripNet(net, "#") # removes comments (lines starting with '#') and entirely empty lines '\n'
    
    try: # when the circuit is not defined in the right <tags>, the user is told and the program exits
        circuitBlock = extractBlock(strippedNet, "<CIRCUIT>", "</CIRCUIT>") # Returns the content between the tags
    except Exception as err:
        print(err, " when extracting Circuit Block from net.\nExiting program.")
        exit()
    try:
        termsBlock = extractBlock(strippedNet, "<TERMS>", "</TERMS>")
    except Exception as err:
        print(err, " when extracting Terms Block from net.\nExiting program.")
        exit()
    try:
        outputBlock = extractBlock(strippedNet, "<OUTPUT>", "</OUTPUT>")
    except Exception as err:
        print(err, " when extracting Output Block from net.\nExiting program.")
        exit()

    circConditions = defineConditions(termsBlock) # define conditions like VT and RS in a dictionary with the key given in the term block

    print("Calculate Frequencies to Simulate...")
    testFreqs = findFrequencies(circConditions) # Calculate the frequencies to test at

    print("Sorting Elements...")
    sortedElements = sortR(circuitBlock) # Returns the components ordered by the appearance in the circuit, by its input node number

    print("Checking for str valued components/nodes...") # In case of strings being assigned to component values
    for comp in sortedElements:
        try:
            float(comp[2].split("=")[-1])
        except ValueError: # Invalid assingments in the net
            exit()

    print("Converting Impedances to Resistances...")
    allFreqImpedances = []
    for i in testFreqs: # for each test frequency, find the impedances of each component 
        allFreqImpedances.append(convertImpedances(sortedElements,i))

    print("Computing all cascade matrices...")
    matrices = [np.array([[1,0],[0,1]],dtype='complex128')]*len(allFreqImpedances)
    for count,frequency in enumerate(allFreqImpedances): # for each test frequency, find the cascade matrix for the circuit's impedances
        matrices[count] = (buildMatrix(frequency))
    
    print("Extracting Outputs...")
    variablesToFind = extractOutputs(outputBlock) # Look through the output block and find the order and names of the output variables

    print("Calculating all output values...")
    variablesDict = calculateOutputs(variablesToFind, circConditions, matrices) # Calculate the output variables, returns a dictionary with keys of the variables, assigned a list of values for each test frequency
    
    print("Building and writing the output file...")
    outFileName = writeOutputFile(variablesDict,variablesToFind, testFreqs,args[2]) # Returns the .csv file name, but writes the data to a file that is formatted meticulously to match the models

    print("DONE!") # If all is successful (excecuting without errors), the code with print and then exit cleanly
    
    return 0 # Convention carried from C denoting no errors while excecuting


# Returns the contents of the net file, and closes it to reduce memory use and increase stability
def importNet(file):
    print("Starting import of net...")
    try :
        netFile = open(file, 'r')
        netData = netFile.readlines()
        netFile.close()
        print("Net Importing Successful!")
        # print(netData) # for testing
        return netData
    except:
        print("Error opening net file!")
        return 1 # Convention from C denoting an error occured


# removes comments (lines starting with '#') and entirely empty lines '\n'
def stripNet(NET, sym):
    print("Stripping Comments and empty lines (\\n) from net...")
    stripped = ""
    for line in NET: # Keeps line that are not empty newlines and aren't comments indicated by the sym variable
        if (sym not in line) and (line[0] != '\n'):
            stripped += line
    print("Net stripped")
    return stripped


# Returns the content between the tags
def extractBlock(file, pre, post):
    print("Extracting {}...".format(pre[1:-1])) # removes the '<' and '>' characters
    return (file.split(pre)[1]).split(post)[0] 
""" By returning the result of the extraction directly, 
    the interpreter can optimise the execution better 
    and not need to assign it to a temporary variable.
"""


# Define conditions like VT and RS in a dictionary with the keys and values given in the term block
def defineConditions(block):
    cacheDict = {} # makes an empty dictionary variable
    expressions = block.split() # turn the str into a list of lines
    for value in expressions:
        express = value.split('=') # get the values defined after the '='
        cacheDict[express[0]] = float(express[1]) # adds the conditions to the dictionary
    return cacheDict


# Calculate the frequencies to test at
def findFrequencies(conditions):
    try: # checks for linear spread of frequencies
        start = float(conditions["Fstart"])
    except KeyError:
        try: # check for logarithmic spread of frequencies
            start = float(conditions["LFstart"])
        except KeyError:
            print("Start Frequency not given, or in wrong format. Exiting...")
            exit()
    try:# checks for linear spread of frequencies
        end = float(conditions["Fend"])
    except KeyError:
        try: # check for logarithmic spread of frequencies
            end = float(conditions["LFend"])
        except KeyError:
            print("End Frequency not given, or in wrong format. Exiting...")
            exit()
    try:
        df = float( (end-start) / (float((conditions["Nfreqs"]))-1)) # to make sure that the difference doesn't go beyond the bounds
    except KeyError: # if the number of frequencies is not given, exit
        exit()
    try:
        if "Fstart" in conditions: # linear spread
            test_Freqs = np.arange(start=start, stop=end+1, step=df) # makes an array of evenly spaced (df) frequencies
        elif "LFstart" in conditions: # logarithmic spread
            test_Freqs = np.logspace(np.log10(start),np.log10(end), num=int(conditions["Nfreqs"]), base=10) # array of frequencies by log space
        else:
            raise KeyError
    except Exception as err:
        print("Error: ", err, ", in testFreqs assignment.\nExiting program.")
        exit()
    return test_Freqs


# Returns the components ordered by the appearance in the circuit, by its input node number
def sortR(circuit):
    elements = []
    circuit = circuit.split('\n')[1:-1] # Change the string into a list of strings
    for elem in circuit:
        position = int(elem.split('n1=')[1].split(' ',1)[0]) # Gets the value of n1
        output = int(elem.split('n2=')[1].split(' ',1)[0]) # Gets the value of n2
        resistance = str(elem.split(' ')[-1]) # Gets the value of component
        elements.append(list((position,output,resistance))) # Makes a list(ordered & changeable) of tuples(ordered & unchangeable)
    elements.sort() # Sorts by the input node connection, position
    return elements


# Find the impedance of each component, while maintaining the complex nature of the numbers. This function can be optimised a lot
def convertImpedances(blockRef, currentFreq): # could do the calculations in 2 1d matrixes
    omega = 2.0 * np.pi * float(currentFreq) * 1j # defining the frequency in rad/s
    impedance = ['R','L','C','G']
    block = [[None]*3] * len(blockRef) # Initialise the list with the required lengths and dimensions
    for count,component in enumerate(blockRef):
        x = ""
        for imp in impedance:
            if imp in component[2]: # Search for the type of component
                x = imp
                break
        
        xf = complex(component[2].split('=')[-1]) # gets the component value
        z = 0.0+0j # initialise as a complex number
        if x == 'R': # switch...case could be used, but may be optimised by the interpreter
            z = xf # Resistor impedanceis frequency independant
        elif x == 'L':
            z = omega * xf # Inductor impedance equation
        elif x == 'C':
            if omega != 0j: # checks for not a dc voltage
                z = 1 / (omega * xf) # Capacitor impedance equation
            else:
                z = complex('inf') # divide by 0 error handling
        elif x == 'G':
            z = 1 / xf # Conductance impedance conversion

        temp = [blockRef[count][0],blockRef[count][1],z]
        block[count] = temp # inefficient but somewhat necessary
    return block
"""
    In testing, there was an issue of the list filling with identical information
    The implementation fixed the issue, though the method did work sometimes
    Maybe a change in environment/interpreter caused this
    The problem method is included here for interested individuals
    -------------------------------------------------
    newLine = line.split(x)[0] + "R=" + str(z)
    block[count][0:1] = blockRef[count][0:1] 
    block[count][2] = z
"""


# Find the cascade matrix for the circuit's impedances. Using easy parallisation () did not improve the speed
def buildMatrix(components): # takes in all components in input order and combines the impedances
    cascadeMatrix = np.array([[1,0],
                              [0,1]], dtype='complex128')
    for resistor in components:
        if resistor[1] != 0: # series
            rMatrix = np.array([[1,(resistor[2])],
                                [0,            1]], dtype='complex128')
        else: # parallel
            if resistor[2] != 0:
                rMatrix = np.array([[ 1,             0], # Constructs the matrix to be multiplied to the previous cascade matrix
                                    [(1/resistor[2]),1]], dtype='complex128')
            else:
                rMatrix = np.array([[ 1,    0], # checks for 0 impedance to ground
                                    [np.inf,1]], dtype='complex128')
        cascadeMatrix = np.dot(cascadeMatrix, rMatrix) # multiplies cascade matrix with component matrix, stores result in cascade matrix 
    return cascadeMatrix
"""
    numpy.dot was faster than numpy.matmul, 672ms compared to matmul with 786ms for e_Ladder_400
    numpy.linalg.multi_dot was tested, but was orders of magnitude slower, taking as long for one frequency as numpy.dot was for all frequencies
    Parallel processing was tested (concurrent.futures.ThreadPoolExcecuter), but the program ran slower (1.39s with vs 1s without the overhead of parallel processing)
"""


# Look through the output block and find the order and names of the output variables
def extractOutputs(file):
    variable_unit = []
    lines = file.split('\n') # string to list of lines
    for line in lines[1:-1]: # get rid of the <tag> lines
        variable_unit.append(line.split(' ', 1)) # Separates variable and unit
    for count,i in enumerate(variable_unit): # adds L for the unit for V and C amplification
        if 'A' in variable_unit[count][0]:
            if len(i) == 1:
                variable_unit[count].append('L')
            elif len(i) == 2:
                variable_unit[count][1] = 'L'
        elif len(variable_unit[count][1].split(' ')) != 1:
            tempv_u = variable_unit[count][1].split(' ')
            for i in tempv_u:
                if len(i) != 0:
                    variable_unit[count][1] = i
                    break
        
    return variable_unit


# Calculate the output variables, returns a dictionary with keys of the variables, assigned a list of values for each test frequency
def calculateOutputs(probes = list(), circCond = dict(), matrix = list()):
    potentialValues = {"Vin" : 0j, #
                       "Iin" : 0j, #
                       "Pin" : 0j, #
                       "Zin" : 0j, #
                       "Vout" : 0j, #
                       "Iout" : 0j, #
                       "Pout" : 0j, #
                       "Zout" : 0j, #
                       "Av" : 0j, #
                       "Ai" : 0j #
                       }
    outputVariables = {}
    for p in probes[:]:
        outputVariables[p[0]] = []
    
    # Calculation order is very important as some require other values to be already known
    for i in range(len(matrix)):
        [[Ac,Bc],[Cc, Dc]] = matrix[i] # sets variables to the values in the matrix, to be easier to read the code
        Zl = circCond["RL"]

        if "RS" in circCond: # get the source resistance
            Zs = circCond["RS"]
        else:
            Zs = 1 / circCond["GS"]


        

        potentialValues["Zin"] = complex(((Ac*Zl) + Bc) / ((Cc*Zl)+Dc))
        potentialValues["Zout"] = complex(((Dc*Zs) + Bc) / ((Cc*Zs)+Ac))

        potentialValues["Av"] = complex(Zl / ((Ac*Zl)+Bc))
        potentialValues["Ai"] = complex(1 / ((Cc*Zl)+Dc))

        if "VT" in circCond: # Thevenin voltage source with series resistance
            Vs = circCond["VT"]
            if "RS" in circCond:
                Zs = circCond["RS"]
            else:
                Zs = 1 / circCond["GS"]
            potentialValues["Vin"] = complex(Vs*(potentialValues["Zin"] / (Zs+potentialValues["Zin"]))) # Vin first
            potentialValues["Iin"] = complex(potentialValues["Vin"] / potentialValues["Zin"])
        elif "IN" in circCond: # Norton current source with parallel resistance
            Is = circCond["IN"]
            if "RS" in circCond:
                Zs = circCond["RS"]
            else:
                Zs = 1 / circCond["GS"]
            potentialValues["Iin"] = complex(Is*(potentialValues["Zin"] / (Zs+potentialValues["Zin"]))) # Iin first
            potentialValues["Vin"] = complex(Is*((Zs*potentialValues["Zin"]) / (Zs+potentialValues["Zin"])))
        
        potentialValues["Pin"] = potentialValues["Vin"] * np.conjugate(potentialValues["Iin"])

        potentialValues["Iout"] = potentialValues["Vin"] / (Ac*Zl + Bc) # Need to prove from base definitions
        potentialValues["Vout"] = potentialValues["Iout"] * Zl # V = I R
        potentialValues["Pout"] = potentialValues["Vout"] * np.conjugate(potentialValues["Iout"])
        
        for p in probes:
            outputVariables[p[0]].append(potentialValues[p[0]])
    
    return outputVariables
"""
    The documentation provided suggested the below method to be valid, 
    however, at low frequencies (10Hz), the values are incorrect. 
    The implemented method is correct and is based on the relationships of linear circuits.

    try:
            invM = np.linalg.inv(matrix[i]) # can take in multiple matrixes
    except np.linalg.LinAlgError: # when singularity matrix, use pseudo-inversion
        invM = np.linalg.pinv(matrix[i])
        print("Used psudeo-inversion due to singularity matrix")

    [potentialValues["Vout"],potentialValues["Iout"]] = np.matmul(invM, np.array([potentialValues["Vin"],potentialValues["Iin"]]))
"""


# Returns the .csv file name, but writes the data to a file that is formatted meticulously to match the model files
def writeOutputFile(vars = dict(), vAndUnits = list(), freqs = list(), filename = str()):    
    print("Formatting output file...")
    f = open(filename, 'a') # append mode so it doesn't overwrite from here on
    
    units = dict(vAndUnits) # makes looking up the units for a given variable easier in this implementation

    # 3 Variables are used to build each section of the file
    variableStr = "{spaces}Freq,".format(spaces = " "*6)
    unitStr = "{spaces}Hz,".format(spaces = ' '*8)
    valueStr = ""

    # Adding the Re and Im prefix indicators to the variables
    for i in vars:
        variableStr += "{spaces}{comp}({var}),".format(spaces=' '*(7-len(i)), comp = "Re", var=i)
        variableStr += "{spaces}{comp}({var}),".format(spaces=' '*(7-len(i)), comp = "Im", var=i)
    variableStr = variableStr[:-1] + '\n' # removes the last comma and adds a new line

    for i in units:
        unitStr += "{spaces}{unit},".format(spaces=' '*(11-len(units[i])), unit=(units[i]))
        unitStr += "{spaces}{unit},".format(spaces=' '*(11-len(units[i])), unit=(units[i]))
    unitStr = unitStr[:-1] # remove the last comma
    unitStr += '\n'

    for fcount in range(len(freqs)):
        valueStr += " {j:.3e},". format(j = freqs[fcount]) # sets floats to be in scientific notation with 4 significant figures, then converts to a string
        for var in vars:
            freqVals = vars[var][fcount]
            realPart = np.real(freqVals)
            imagPart = np.imag(freqVals)

            if (realPart > 0): # maintain the width of the columns with this spacing, accounts for negative sign. math.copysign is needed due to -0.00 is seen as 0.00 so formatting messes up
                polarity = "  " # spacing used for having no sign
            elif (realPart < 0): # if negative and not -0.00
                polarity = ' ' # spacing for having the negative sign
            else:
                polarity = "  "
                realPart = +0.00
            valueStr += "{spaces}{var:.3e},".format(spaces=polarity, var=realPart)
            if (imagPart > 0):
                polarity = "  "
            elif imagPart < 0:
                polarity = ' '
            else:
                polarity = "  "
                imagPart = +0.00
            valueStr += "{spaces}{var:.3e},".format(spaces=polarity, var=imagPart)
        valueStr += '\n'
    
    print("Writing to file...") # So it is known if the program crashes before the writing or during
    f.write(variableStr + unitStr + valueStr) # writes all the data in the specified CSV formatting
    f.close() # closes the file, saving it

    return filename


# Only runs the main function if this is excecuted file, deosn't if this is imported
if __name__ == "__main__": # Run main function when file is run directly
    main(argv)
    # profile.run('main(argv)')