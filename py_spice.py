# PySPICE
# Take in a file with the same format at spice
# Identify probe points
# Convert to ABCD Matricies
# Convert to Impedance
# Calculate propogating 2in-2out voltage/current/frequency at each time step
# Work for DC systems and frequency response
# Output file with results
# Draw circuit and print out

# Net Scraper : X is connected P/S to Y 
# calculating values are probe points
# outputting data to file

# use as many numpy functions as it can operate on matricies efficiently
# make functions for all tranformations and conversions
# use caching ?


##### GPU ACCELERATION WOULD BE FUN

# values could be blank in test files

# next steps
# get the right answers to the values
# using the small circuit and set values, testing with imported data
# could be in build matrix or in convert impedance

# convert the list of nodes to an array with length 4 + Fn
# so the deepcopy doesn't run everytime / ever


# Vout Iout and Pout are wrong for 10Hz
# Vout and Iout are slightly wrong for 10Hz a1
# https://text-compare.com/send_comparison/





# from numba import njit
# import cProfile as profile
from sys import argv # for argv
# import matplotlib.pyplot as plt # Plotting the data
import numpy as np # Arrays and matrix calculations
# import copy # copy.deepcopy() for list of lists


def main(args = []):
    try:
        print("Starting Py_SPICE Program using:\nNet File = {}\nResults File = {}".format(args[1], args[2]))
    except Exception as err:
        print(err, ": Argument 1 must be the net file, Argument 2 must be the output file.")
        print("Arguments given ", args)
        exit()
        
    try:
        open(args[2], 'x').close() # if file doesn't exist it will make it
        print("Creating output file\"{file}\"".format(file=args[2]))
    except:
        open(args[2], 'w').close() # if it does exist it will erase the contents
        print("Erasing output file contents...")
    
    net = importNet(args[1])

    strippedNet = stripNet(net, "#") # removes lines with # and empty lines '\n'
    
    try:
        circuitBlock = extractBlock(strippedNet, "<CIRCUIT>", "</CIRCUIT>") # using split on <> </> then get the 3 bits as different variables
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

    circConditions = defineConditions(termsBlock) # define conditions like VT and RS

    # if 'Nfreqs' in circConditions:
    print("Calculate Frequencies to Simulate...")
    try:
        start = float(circConditions["Fstart"])
    except KeyError:
        # print("Start frequency not given, using 0Hz")
        # start = 0
        exit()
    try:
        end = float(circConditions["Fend"])
    except KeyError:
        # print("End frequency not given, using 0Hz")
        # end = 0
        exit()
    try:
        df = float( (end-start) / (float((circConditions["Nfreqs"]))-1)) # check if right
    except KeyError:
        # print("Number of test frequencies not given, using 1")
        # df = 1
        # # print(df)
        exit()
    try:
        testFreqs = np.arange(start=start, stop=end+1, step=df)#np.logspace(np.log10(start),np.log10(end), num=int(circConditions["Nfreqs"]), base=10) # makes an array of evenly spaced (df) frequencies
        for i in testFreqs:
            print(int(i))
    except Exception as err:
        print("Error: ", err, ", in testFreqs assignment.\nExiting program.")
        exit()

    print("Sorting Elements...")
    sortedElements = sortR(circuitBlock)

    print("Removing str valued components/nodes...")
    for count,comp in enumerate(sortedElements):
        try:
            float(comp[2].split("=")[-1])
        except ValueError: # a_Test_Circuit_1BRX
            # print("Removing bad assinment value of component: ", str(comp))
            # sortedElements.pop(count)
            exit()

    print("Converting Impedances to Resistances...")
    allFreqImpedances = []
    for i in testFreqs:
        # copyofsortedElements = []
        # copyofsortedElements.extend(sortedElements) # copy.deepcopy(sortedElements)
        allFreqImpedances.append(convertImpedances(sortedElements,i))

    print("Computing all cascade matrices...")
    matrices = []
    for frequency in allFreqImpedances:
        matrices.append(buildMatrix(frequency))

    print("Extracting Outputs...")
    variablesToFind = extractOutputs(outputBlock) # order and names of variables to report

    print("Calculating all output values...")
    variablesDict = calculateOutputs(variablesToFind, circConditions, matrices)
    
    print("Building and writing the output file...")
    outFileName = writeOutputFile(variablesDict,variablesToFind, testFreqs,args[2])

    print("DONE!")
    
    return 0


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
        return 1


def stripNet(NET, sym):
    print("Stripping Comments and empty lines (\\n) from net...")
    stripped = ""
    for line in NET:
        if (sym not in line) and (line[0] != '\n'):
            stripped += line
    print("Net stripped")
    return stripped


def extractBlock(file, pre, post):
    print("Extracting {}...".format(pre[1:-1]))
    return (file.split(pre)[1]).split(post)[0]


def defineConditions(block):
    cacheDict = {}
    expressions = block.split()
    for value in expressions:
        express = value.split('=')
        cacheDict[express[0]] = float(express[1]) # adds the conditions to the dictionary
    return cacheDict


def sortR(circuit):
    elements = []
    # circuit = circuit.replace('\n','')
    circuit = circuit.split('\n')[1:-1]
    for elem in circuit:
        position = int(elem.split('n1=')[1].split(' ',1)[0]) # Gets the value of n1
        output = int(elem.split('n2=')[1].split(' ',1)[0]) # Gets the value of n2
        resistance = str(elem.split(' ')[-1]) # Gets the value of component
        elements.append(list((position,output,resistance))) # Makes a list of tuples, ordered & unchangable
    elements.sort()
    return elements # Sorts by the input node connection, position


def convertImpedances(blockRef, currentFreq): # could do the calculations in 2 1d matrixes
    omega = float(2) * np.pi * float(currentFreq) 
    impedance = ['R','L','C','G']
    # block = copy.deepcopy(blockRef)
    block = [[None]*3] * len(blockRef)
    for count,component in enumerate(blockRef):
        x = ""
        for imp in impedance:
            if imp in component[2]:
                x = imp
                break
        
        xf = complex(component[2].split('=')[-1]) # gets the component value
        z = 0.0+0j
        if x == 'R':
            z = xf
        elif x == 'L':
            z = 1j * omega * xf
        elif x == 'C':
            if omega != 0j:
                z = -1j / ( omega * xf )
            else:
                z = complex('inf') # divide by 0 error handling
        elif x == 'G':
            z = 1 / xf

        temp = [blockRef[count][0],blockRef[count][1],z]
        block[count] = temp # inefficient but somewhat necessary, see below
        # newLine = line.split(x)[0] + "R=" + str(z)
        # block[count][0:1] = blockRef[count][0:1] #### In testing there was an issue of the leist filling with identical information, this fixed the issue, though the method did work sometimes, maybe a change in environment/interpreter cause this
        # block[count][2] = z
    return block


def buildMatrix(components): # takes in all components in input order and conditions; returns cascaded matrix for the circuit
    cascadeMatrix = np.array([[1,0],
                              [0,1]], dtype='complex128')
    for resistor in components:
        if resistor[1] != 0: # series
            rMatrix = np.array([[1,(resistor[2])],
                                [0,            1]], dtype='complex128')
            cascadeMatrix = cascadeMatrix @ rMatrix # multiplies cascade matrix with resistor matrix, stores result in cascade matrix
        else: # parallel
            try:
                rMatrix = np.array([[ 1,             0],
                                    [(1/resistor[2]),1]], dtype='complex128')
            except ZeroDivisionError:
                rMatrix = np.array([[ 1,            0],
                                    [(float('inf')),1]], dtype='complex128')
            cascadeMatrix = cascadeMatrix @ rMatrix # multiplies cascade matrix with resistor matrix, stores result in cascade matrix
    return cascadeMatrix


def extractOutputs(file):
    variable_unit = []
    lines = file.split('\n')
    for line in lines[1:-1]:
        variable_unit.append(line.split(' ', 1)) # removes all spaces
    for count,i in enumerate(variable_unit):
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


def calculateOutputs(probes = list(), circCond = dict(), matrix = list()): # have to deal with Zin first to find others?
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
    
    for i in range(len(matrix)):
        [[Ac,Bc],[Cc, Dc]] = matrix[i]
        Zl = circCond["RL"]

        if "RS" in circCond:
            Zs = circCond["RS"]
        else:
            Zs = 1 / circCond["GS"]

        potentialValues["Zin"] = complex(((Ac*Zl) + Bc) / ((Cc*Zl)+Dc))
        potentialValues["Zout"] = complex(((Dc*Zs) + Bc) / ((Cc*Zs)+Ac))

        potentialValues["Av"] = complex(Zl / ((Ac*Zl)+Bc))
        potentialValues["Ai"] = complex(1 / ((Cc*Zl)+Dc))

        if "VT" in circCond:
            Vs = circCond["VT"]
            if "RS" in circCond:
                Zs = circCond["RS"]
            else:
                Zs = 1 / circCond["GS"]
            potentialValues["Vin"] = complex(Vs*(potentialValues["Zin"] / (Zs+potentialValues["Zin"])))
            potentialValues["Iin"] = complex(potentialValues["Vin"] / potentialValues["Zin"])
        elif "IN" in circCond:
            Is = circCond["IN"]
            if "RS" in circCond:
                Zs = circCond["RS"]
            else:
                Zs = 1 / circCond["GS"]
            potentialValues["Iin"] = complex(Is*(potentialValues["Zin"] / (Zs+potentialValues["Zin"]))) # Iin = Is * Zs//Zin
            potentialValues["Vin"] = complex(Is*((Zs*potentialValues["Zin"]) / (Zs+potentialValues["Zin"])))
        
        potentialValues["Pin"] = potentialValues["Vin"] * np.conjugate(potentialValues["Iin"])

        try:
            invM = np.linalg.inv(matrix[i]) # can take in multiple matrixes
        except np.linalg.LinAlgError: # when singularity matrix, use pseudo-inversion
            invM = np.linalg.pinv(matrix[i])
            print("Used psudeo-inversion due to singularity matrix")

        [potentialValues["Vout"],potentialValues["Iout"]] = np.matmul(invM, np.array([potentialValues["Vin"],potentialValues["Iin"]]))
        potentialValues["Pout"] = potentialValues["Vout"] * np.conjugate(potentialValues["Iout"])
        
        for p in probes:
            outputVariables[p[0]].append(potentialValues[p[0]])
    
    return outputVariables
            

def writeOutputFile(vars = dict(), vAndUnits = list(), freqs = list(), filename = str()):    
    print("Formatting output file...")
    f = open(filename, 'a') # append mode so it doesn't overwrite from here on

    # for count,i in enumerate(vAndUnits):
    #     if i[0] == 'Av' or i[0] == 'Ai':
    #         if len(i) == 1:
    #             vAndUnits[count].append('L')
    #         elif len(i) == 2:
    #             vAndUnits[count][1] = 'L'
    units = dict(vAndUnits)

    variableStr = "{spaces}Freq,".format(spaces = " "*6)
    unitStr = "{spaces}Hz,".format(spaces = ' '*8)
    valueStr = ""

    for i in vars:
        variableStr += "{spaces}{comp}({var}),".format(spaces=' '*(7-len(i)), comp = "Re", var=i)
        variableStr += "{spaces}{comp}({var}),".format(spaces=' '*(7-len(i)), comp = "Im", var=i)
    variableStr = variableStr[:-1] + '\n'

    for i in units:
        unitStr += "{spaces}{unit},".format(spaces=' '*(11-len(units[i])), unit=(units[i]))
        unitStr += "{spaces}{unit},".format(spaces=' '*(11-len(units[i])), unit=(units[i]))
    unitStr = unitStr[:-1]
    unitStr += '\n'

    for fcount in range(len(freqs)):
        valueStr += " {j:.3e},". format(j = freqs[fcount])
        for var in vars:
            freqVals = vars[var][fcount]
            if (np.real(freqVals) >= 0): # maintain the width of the columns with this spacing, accounts for negative sign
                polarity = "  "
            else:
                polarity = ' '
            valueStr += "{spaces}{var:.3e},".format(spaces=polarity, var=np.real(freqVals))
            if (np.imag(freqVals) >= 0):
                polarity = "  "
            else:
                polarity = ' '
            valueStr += "{spaces}{var:.3e},".format(spaces=polarity, var=np.imag(freqVals))
        # valueStr = valueStr[:-1]
        valueStr += '\n'
    
    print("Writing to file...")
    f.write(variableStr + unitStr + valueStr) # writes all the data in the specified CSV formatting
    f.close() # closes the file, saving it

    return filename


if __name__ == "__main__": # Run main function
    # profile.run('main(sys.argv)')
    main(argv)

#--------------------------------------------^-------^------------------------------------^-----------^-----------------------------------------------------------^-----------^