import ROOT as root
from ROOT import gROOT
import sys, os, click, time
from rootpy import stl
from rootpy.io import File
from rootpy.tree import Tree, TreeModel, TreeChain
import neriX_config, neriX_datasets, neriX_pmt_gain_corrections
import threading
import numpy as np
from rootpy.io.pickler import dump, load
from contextlib import contextmanager
import rootpy.compiled as C
import root_numpy

C.register_file('neriX_gain_correction.C', ['GetGainCorrectionBottomPMT', 'GetGainCorrectionErrorBottomPMT'])


# MUST INCLUDE TYPES TO AVOID SEG FAULT
stl.vector(stl.vector('float'))
stl.vector(stl.vector('int'))

# suppress stdout temporarily with the following
# use is the following:
# with suppress_stdout():
#	function_call_here()
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def success_message(message):
	click.echo(click.style('\n\nSUCCESS: %s\n\n' % message, fg='green', bold=True))



def warning_message(message):
	click.echo(click.style('\n\nWARNING: %s\n\n' % message, fg='yellow', bold=True, blink=False))


def failure_message(message):
	click.echo(click.style('\n\nFAILURE: %s\n\n' % message, fg='red', bold=True))



def debug_message(message):
	click.echo(click.style('\n\nDEBUG: %s\n\n' % message, fg='white', bold=True))



def get_gain_correction_err(run, unixtime):
	return C.GetGainCorrectionErrorBottomPMT(run, unixtime)




def get_gain_correction(run, unixtime):
	return C.GetGainCorrectionBottomPMT(run, unixtime)




def pull_all_files_given_parameters(run, anodeSetting, cathodeSetting, degreeSetting):
	lFilesToLoad = []
	dDatasets = neriX_datasets.run_files
	
	degreeIndex = 2
	
	for file in dDatasets[run]:
		lParameters = list(dDatasets[run][file])
		if isinstance(lParameters[degreeIndex], dict):
			lParameters[degreeIndex] = lParameters[degreeIndex].keys()
			if lParameters[0] == anodeSetting and lParameters[1] == cathodeSetting and degreeSetting in lParameters[degreeIndex]:
				lFilesToLoad.append(file)
		else:
			if lParameters == [anodeSetting, cathodeSetting, degreeSetting]:
				lFilesToLoad.append(file)
	
	return lFilesToLoad



def print_info_for_processing_multiple_files(run, lAnodeSettings=None, lCathodeSettings=None, lDegreeSettings=None, numProcessors = 20):
	
	if lAnodeSettings == None:
		lAnodeSettings = [4.500]
	elif isinstance(lAnodeSettings, str):
		lAnodeSettings = [lAnodeSettings]

	if lCathodeSettings == None:
		lCathodeSettings = [0.345, 0.700, 1.054, 1.500, 2.356]
	elif isinstance(lCathodeSettings, str):
		lCathodeSettings = [lCathodeSettings]

	if lDegreeSettings == None:
		lDegreeSettings = [-1, -2, -4, -5, -6, -10]
	elif isinstance(lDegreeSettings, str):
		lDegreeSettings = [lDegreeSettings]

	lFilesToLoad = []

	for anodeSetting in lAnodeSettings:
		for cathodeSetting in lCathodeSettings:
			for degreeSetting in lDegreeSettings:
				lFilesToLoad += pull_all_files_given_parameters(run, anodeSetting, cathodeSetting, degreeSetting)

	lFilesToLoad.sort()

	#for i in xrange(len(lFilesToLoad)):
	#	lFilesToLoad[i] = lFilesToLoad[i][:-5]

	d_xml_files = {(16, -1):'nerix-cs-na-low-voltages-run_16.xml',
				   (16, -5):'nerix-cs-na-low-voltages-run_16.xml',
				   (16, -6):'nerix-bkg-low-voltages-run_15.xml',
				   (16, -4):'nerix-default-run_16.xml',
				   (16, -2):'nerix-bkg-low-voltages-run_15.xml',
				   (16, -10):'nerix-gas-gain-run_16.xml'}

	s_for_processing_script = ''
	for file in lFilesToLoad:
		degreeSetting = neriX_datasets.run_files[run][file][2]
		s_for_processing_script += 'call(["./scripts/process.py", "-f", "%s", "%s", "%d"])\n' % (d_xml_files[(run, degreeSetting)], file[:-5], numProcessors)

	print s_for_processing_script



def save_plot(lDirectories, canvas, filename, lFileTypes=['png', 'C'], batch_mode=False):
	# arguments should be in the following form:
	# path_to_image = ./lDirectories[0]/lDirectories[1]/<filename>.<fileType>
	
	if not batch_mode:
		print '\n'
		response = raw_input('Would you like to save the canvas to a file?  If so, please enter "y" otherwise press enter: ')
		print '\n'
		
		if response != 'y':
			return

	if not isinstance(lDirectories, str):
		sPath = './'
		for directory in lDirectories:
			sPath += '%s/' % directory
	else:
		sPath = lDirectories
		
	# check if path exists and make it if not
	if not os.path.exists(sPath):
		os.makedirs(sPath)

	for type in lFileTypes:
		canvas.Print('%s%s.%s' % (sPath, filename, type))



def write_root_object(lDirectories, object, filename, batch_mode=False):
	# arguments should be in the following form:
	# path_to_image = ./lDirectories[0]/lDirectories[1]/<filename>.p
	
	if not batch_mode:
		print '\n'
		response = raw_input('Would you like to save %s to a file?  If so, please enter "y" otherwise press enter: ' % str(object))
		print '\n'
		
		if response != 'y':
			return
	
	sPath = './'
	for directory in lDirectories:
		sPath += '%s/' % directory
		
	# check if path exists and make it if not
	if not os.path.exists(sPath):
		os.makedirs(sPath)

	try:
		#dump(object, '%s%s.root' % (sPath, filename))
		fObject = File('%s%s.root' % (sPath, filename), 'recreate')
		object.Write()
		fObject.Close()
	except:
		print 'ERROR: Pickling object of type %s failed.' % str(type(object))



def check_for_file_info(nameOfFile):
	if not nameOfFile[-5:] == '.root':
		nameOfFile += '.root'

	for run in neriX_datasets.runsInUse:
		try:
			tParameters = neriX_datasets.run_files[run][nameOfFile]
			return tParameters
		except:
			pass



def convert_name_to_unix_time(fileName):
	if fileName[-5:] == '.root':
		fileName = fileName[:-5]
	
	# check that we know position of dates in name
	if fileName[0:5] == 'nerix':
		sTimeTaken = '20' + fileName[6:17] + '00'
		return int(time.mktime(time.strptime(sTimeTaken, '%Y%m%d_%H%M%S')))
	elif fileName[0:8] == 'ct_nerix':
		sTimeTaken = '20' + fileName[9:20] + '00'
		return int(time.mktime(time.strptime(sTimeTaken, '%Y%m%d_%H%M%S')))
	else:
		print 'Only know how to handle nerix and ct_nerix files currently'
		print 'Please add to convert_name_to_unix_time function in order to handle the new case'
		sys.exit()



# profiles a 2d histogram using the median
# rather than the mean
def profile_y_median(hist_to_profile, percent_from_median=34.1):
	num_bins_x = hist_to_profile.nbins(0)
	num_bins_y = hist_to_profile.nbins(1)
	
	bounds_x = hist_to_profile.bounds(0)
	bounds_y = hist_to_profile.bounds(1)
	
	step_size_x = (bounds_x[1] - bounds_x[0]) / float(num_bins_x)
	step_size_y = (bounds_y[1] - bounds_y[0]) / float(num_bins_y)
	
	bin_centers_x = np.linspace(bounds_x[0] + step_size_x/2., bounds_x[1] - step_size_x/2., num=num_bins_x)
	bin_centers_y = np.linspace(bounds_y[0] + step_size_y/2., bounds_y[1] - step_size_y/2., num=num_bins_y)

	aHist = root_numpy.hist2array(hist_to_profile, include_overflow=False)

	# outer loop are columns (what we want)!
	# inner array is small to large
	
	aXErrLow = np.asarray([step_size_x/2. for i in xrange(num_bins_x)])
	aXErrHigh = np.asarray([step_size_x/2. for i in xrange(num_bins_x)])
	
	aYValues = np.zeros(num_bins_x)
	aYErrLow = np.zeros(num_bins_x)
	aYErrHigh = np.zeros(num_bins_x)

	lIndicesToDelete = []
	for slice_number, x_slice in enumerate(aHist):
		points_in_slice = np.zeros(int(np.sum(x_slice)))
		index_counter = 0
		for i, bin_count in enumerate(x_slice):
			for j in xrange(bin_count):
				points_in_slice[index_counter] = bin_centers_y[i]
				index_counter += 1
		if index_counter > 2:
			aYValues[slice_number] = np.median(points_in_slice)
			low_err, high_err = np.percentile(points_in_slice, [50-percent_from_median, 50+percent_from_median])
			aYErrLow[slice_number], aYErrHigh[slice_number] = aYValues[slice_number]-low_err, high_err-aYValues[slice_number]
		else:
			lIndicesToDelete.append(slice_number)

	# delete points that are "empty"
	num_bins_x -= len(lIndicesToDelete)
	bin_centers_x = np.delete(bin_centers_x, lIndicesToDelete)
	aYValues = np.delete(aYValues, lIndicesToDelete)
	aXErrLow = np.delete(aXErrLow, lIndicesToDelete)
	aXErrHigh = np.delete(aXErrHigh, lIndicesToDelete)
	aYErrLow = np.delete(aYErrLow, lIndicesToDelete)
	aYErrHigh = np.delete(aYErrHigh, lIndicesToDelete)


	return (num_bins_x, bin_centers_x, aYValues, aXErrLow, aXErrHigh, aYErrLow, aYErrHigh)



def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))



def create_canvas_for_multiple_histograms(lHists):
	# lHist is a list of dictionaries that will be
	# called dHists

	# dHists should include the key 'hist' with a Hist
	# or Hist2D
	# if Hist is given, one should also include keyword
	# 'color' since they will be drawn on top of each other

	# optional keys
	#

	numberOfHistograms = len(lHists)




# returns all arguments needed to create TGraphAsymmErrors
# simply use *on the returned set to unpack them
def create_graph_with_confidence_interval_for_fit(graphUsedForFit, virtualFitter, confidence_level=0.68):
	numPoints = graphUsedForFit.GetN()
	
	# grab X values
	aXValues = np.zeros(numPoints, dtype=np.float32)
	aYValues = np.zeros(numPoints, dtype=np.float32)
	bXValuesOriginal = graphUsedForFit.GetX()
	bXValuesOriginal.SetSize(numPoints)
	aXValuesOriginal = np.array(bXValuesOriginal, 'f')
	for i, value in enumerate(aXValuesOriginal):
		aXValues[i] = value
	
	gConfidenceInterval = root.TGraphErrors(numPoints, aXValues, aYValues)
	virtualFitter.GetConfidenceIntervals(gConfidenceInterval, confidence_level)

	# grab Y values
	#print gConfidenceInterval.GetErrorY(5)
	bValues = gConfidenceInterval.GetY()
	bValues.SetSize(numPoints)
	aYValues = np.array(bValues, 'f')

	# need to grab X/Y errors one at a time (stupid root...)
	aXErrLow = np.zeros(numPoints, dtype=np.float32)
	aXErrHigh = np.zeros(numPoints, dtype=np.float32)
	aYErrLow = np.zeros(numPoints, dtype=np.float32)
	aYErrHigh = np.zeros(numPoints, dtype=np.float32)
	for i in xrange(numPoints):
		#print i, gConfidenceInterval.GetErrorYlow(i)
		aXErrLow[i] = gConfidenceInterval.GetErrorXlow(i)
		aXErrHigh[i] = gConfidenceInterval.GetErrorXhigh(i)
		aYErrLow[i] = gConfidenceInterval.GetErrorYlow(i)
		aYErrHigh[i] = gConfidenceInterval.GetErrorYhigh(i)

	#print type(aYErrLow)

	return (numPoints, aXValues, aYValues, aXErrLow, aXErrHigh, aYErrLow, aYErrHigh)



def convert_2D_hist_to_matrix(h2D):
	numBinsX = h2D.nbins(0)
	numBinsY = h2D.nbins(1)
	
	a2D = np.zeros((numBinsX, numBinsY))

	for i in xrange(numBinsX):
		for j in xrange(numBinsY):
			a2D[i,j] = h2D[i+1,j+1].value

	return a2D






class neriX_analysis:
	__GERMANIUM_SLOPE = '693.505789'
	__GERMANIUM_INTERCEPT = '1.717426'

	def __init__(self, lFilesToLoad = None, degreeSetting = None, cathodeSetting = None, anodeSetting = None):
	
		if isinstance(lFilesToLoad, str):
			lFilesToLoad = [lFilesToLoad]
		
		numFiles = len(lFilesToLoad)
		
		if numFiles > 1:
			assert degreeSetting != None
			assert cathodeSetting != None
			assert anodeSetting != None


		PARAMETERS_INDEX = neriX_datasets.PARAMETERS_INDEX
		ANODE_INDEX = neriX_datasets.ANODE_INDEX
		CATHODE_INDEX = neriX_datasets.CATHODE_INDEX
		DEGREE_INDEX = neriX_datasets.DEGREE_INDEX

		dRunFiles = neriX_datasets.run_files
		lRuns = neriX_datasets.runsInUse
		lParameters = None
		
		gROOT.ProcessLine('.L ' + str(neriX_config.pathToModules) + 'neriX_gain_correction.C+')
		# gROOT.ProcessLine('.L ' + str(neriX_config.pathToModules) + 'neriX_pos_correction.C+')

		# check neriX_datasets for file and parameters
		for i in xrange(len(lFilesToLoad)):
			if lFilesToLoad[i][-5:] != '.root':
				lFilesToLoad[i] += '.root'
			
		# just grab neriX portion
		self.lFilenames = []
		for file in lFilesToLoad:
			self.lFilenames.append(file[-22:])
		
		for run in lRuns:
			try:
				lParameters = dRunFiles[run][self.lFilenames[0]]
				currentRun = run
				break
			except KeyError:
				continue
			
		if not lParameters:
			print str(self.lFilenames[0]) + ' does not appear in neriX_datasets.py'
			print 'Please check if the file exists and that your spelling is correct'
			sys.exit()
		
		"""
		if numFiles == 1:
			self.anodeSetting = lParameters[ANODE_INDEX]
			self.cathodeSetting = lParameters[CATHODE_INDEX]
			self.degreeSetting = lParameters[DEGREE_INDEX]
		else:
			self.anodeSetting = anodeSetting
			self.cathodeSetting = cathodeSetting
			self.degreeSetting = degreeSetting
		"""
		self.anodeSetting = lParameters[ANODE_INDEX]
		self.cathodeSetting = lParameters[CATHODE_INDEX]
		self.degreeSetting = lParameters[DEGREE_INDEX]
		self.runNumber = currentRun
		
		
		# make final list of files that match anode, cathode, and degree settings
		self.lRootFiles = []
		# reset filenames to only include good ones
		self.lFilenames = []
		for file in lFilesToLoad:
			try:
				if dRunFiles[self.runNumber][file[-22:]] == (self.anodeSetting, self.cathodeSetting, self.degreeSetting):
					print 'Adding ' + str(file[-22:]) + ' to list.'
			
					if file[0] == 'n' or file[0] == 'c':
						pathToFile = neriX_config.pathToData + 'run_' + str(self.runNumber) + '/' + str(file)
					else:
						pathToFile = file
					
					try:
						self.lRootFiles.append(File(pathToFile, 'read'))
					except:
						print '\n\nCould not locate ' + str(pathToFile)
						if numFiles > 1:
							print 'Continuing without file'
							numFiles -= 1
							continue
						else:
							sys.exit()
					if self.lRootFiles[-1].keys() == []:
						print 'Problem opening file - please check name entered'
						print 'Entered: ' + pathToFile
						self.lRootFiles.pop()
					else:
						print 'Successfully added ' + str(file[-22:]) + '!\n'
						self.lFilenames.append(file[-22:])
				else:
					print 'File ' + str(file[-22:]) + ' does not match set values for the anode, cathode, or degree (skipping).'
			except KeyError:
				print str(file[-22:]) + ' does not appear in neriX_datasets.py'
				print 'Please check if the file exists and that your spelling is correct'
				
		# update number of files
		numFiles = len(self.lRootFiles)
			
		self.lT1 = [0 for i in xrange(numFiles)]
		self.lT2 = [0 for i in xrange(numFiles)]
		self.lT3 = [0 for i in xrange(numFiles)]
		self.lT4 = [0 for i in xrange(numFiles)]
		self.lEList = [0 for i in xrange(numFiles)]
		
		for i, rootFile in enumerate(self.lRootFiles):
			self.lT1[i] = rootFile.T1
			self.lT2[i] = rootFile.T2
			self.lT3[i] = rootFile.T3
			try:
				self.lT4[i] = rootFile.T4
			except:
				self.lT4[i] = None

			self.lT1[i].SetName('T1_' + self.lFilenames[i])
			self.lT2[i].SetName('T2_' + self.lFilenames[i])
			self.lT3[i].SetName('T3_' + self.lFilenames[i])
			if self.lT4[i]:
				self.lT4[i].SetName('T4_' + self.lFilenames[i])

			self.lT1[i].create_buffer()
			self.lT2[i].create_buffer()
			self.lT3[i].create_buffer()
			if self.lT4[i]:
				self.lT4[i].create_buffer()

			self.lT1[i].AddFriend('T2_' + self.lFilenames[i])
			self.lT1[i].AddFriend('T3_' + self.lFilenames[i])
			if self.lT4[i]:
				self.lT1[i].AddFriend('T4_' + self.lFilenames[i])
		
			# create event list
			self.lEList[i] = root.TEventList('eList_' + self.lFilenames[i])
			root.SetOwnership(self.lEList[i], True)
		
			#define helpful aliases
			self.lT1[i].SetAlias('dt','(S2sPeak[0]-S1sPeak[0])/100.')
			self.lT1[i].SetAlias('s1_tot_bot_minus_noise_m','(S1sTotBottom[0] - S1sNoiseMedian[0][16])')
			self.lT1[i].SetAlias('s1_tot_bot_minus_noise_t','(S1sTotBottom[0] - S1sNoiseTrapezoid[0][16])')
			numEJs = 4
			for ejNum in xrange(numEJs):
				self.lT1[i].SetAlias('psd%d' % (ejNum),'LiqSciTailRaw[%d]/LiqSciRaw[%d]' % (ejNum, ejNum))
			
			if self.lT4[i]:
				self.lT1[i].SetAlias('R','sqrt(pow(ctNNPos[0][0],2)+pow(ctNNPos[0][1],2))')
				self.lT1[i].SetAlias('X','ctNNPos[0][0]')
				self.lT1[i].SetAlias('Y','ctNNPos[0][1]')
				self.lT1[i].SetAlias('R_old','sqrt(pow(S2sPosFann[0][0],2)+pow(S2sPosFann[0][1],2))')
				self.lT1[i].SetAlias('X_old','S2sPosFann[0][0]')
				self.lT1[i].SetAlias('Y_old','S2sPosFann[0][1]')
			else:
				self.lT1[i].SetAlias('R','sqrt(pow(S2sPosFann[0][0],2)+pow(S2sPosFann[0][1],2))')
				self.lT1[i].SetAlias('X','S2sPosFann[0][0]')
				self.lT1[i].SetAlias('Y','S2sPosFann[0][1]')


			self.lT1[i].SetAlias('czS1sTotBottom','(1./(0.863098 + (-0.00977873)*Z))*S1sTotBottom')
			
			
			self.lT1[i].SetAlias('s1asym', '(S1sTotBottom[0]-S1sTotTop[0])/S1sTot[0]')
			self.lT1[i].SetAlias('s2asym', '(S2sTotBottom[0]-S2sTotTop[0])/S2sTot[0]')
			
			if self.runNumber == 15 or self.runNumber == 16:
				# Z is taken from the GATE in runs 15 and 16
				self.dt_offset_gate = neriX_datasets.d_dt_offset_gate[self.runNumber]
			
				if self.cathodeSetting == 0.345:
					self.lT1[i].SetAlias('Z','-1.56*(dt-%f)' % self.dt_offset_gate)
				elif self.cathodeSetting == 0.700:
					self.lT1[i].SetAlias('Z','-1.68*(dt-%f)' % self.dt_offset_gate)
				elif self.cathodeSetting == 1.054:
					self.lT1[i].SetAlias('Z','-1.77*(dt-%f)' % self.dt_offset_gate)
				elif self.cathodeSetting == 1.500:
					self.lT1[i].SetAlias('Z','-1.87*(dt-%f)' % self.dt_offset_gate)
				elif self.cathodeSetting == 2.356:
					self.lT1[i].SetAlias('Z','-2.00*(dt-%f)' % self.dt_offset_gate)
				elif self.cathodeSetting == 5.500:
					self.lT1[i].SetAlias('Z','-2.233*(dt-%f)' % self.dt_offset_gate)
				else:
					print 'Incorrect field entered - cannot correct Z'
			elif self.runNumber == 10 or self.runNumber == 11:
				if self.cathodeSetting == 0.345:
					self.lT1[i].SetAlias('Z','-1.511*dt')
				elif self.cathodeSetting == 1.054:
					self.lT1[i].SetAlias('Z','-1.717*dt')
				elif self.cathodeSetting == 2.356:
					self.lT1[i].SetAlias('Z','-1.958*dt')
				elif self.cathodeSetting == 5.500:
					self.lT1[i].SetAlias('Z','-2.208*dt')
				else:
					print 'Incorrect field entered - cannot correct Z'

			#will need to change these constants depending on Ge Calibratio
			self.lT1[i].SetAlias('GeEnergy',self.__GERMANIUM_SLOPE + '*GeHeight[0] + ' + self.__GERMANIUM_INTERCEPT)
			self.lT1[i].SetAlias('eDep','661.657 - GeEnergy')
			
			self.lT1[i].SetAlias('ratio_s2tot1_s2top0','S2sTot[1] / S2sTotTop[0]')
			
			
			# need to create compiled function for ct alias
			if self.runNumber == 10 or self.runNumber == 11:
				self.lT1[i].SetAlias('ctS1sTotBottom','1./(GetGainCorrectionBottomPMT(' + str(self.runNumber) + ', TimeSec))*czS1sTotBottom')
				self.lT1[i].SetAlias('ctS2sTotBottom','1./(GetGainCorrectionBottomPMT(' + str(self.runNumber) + ', TimeSec))*S2sTotBottom')
			
			elif self.runNumber == 16:
				self.lT1[i].SetAlias('ctS1sTotBottom','1./(GetGainCorrectionBottomPMT(' + str(self.runNumber) + ', TimeSec))*S1sTotBottom')
				self.lT1[i].SetAlias('ctS2sTotBottom','1./(GetGainCorrectionBottomPMT(' + str(self.runNumber) + ', TimeSec))*S2sTotBottom')
			
				self.lT1[i].SetAlias('cpS1sTotBottom','( 1. / (8.49e-1 - 1.06e-2*Z) )*ctS1sTotBottom')
				self.lT1[i].SetAlias('cpS2sTotBottom','(1.)*ctS2sTotBottom')
			
			else:
				self.lT1[i].SetAlias('ctS1sTotBottom','1./(GetGainCorrectionBottomPMT(' + str(self.runNumber) + ', TimeSec))*S1sTotBottom')
				self.lT1[i].SetAlias('ctS2sTotBottom','1./(GetGainCorrectionBottomPMT(' + str(self.runNumber) + ', TimeSec))*S2sTotBottom')
			
				self.lT1[i].SetAlias('cpS1sTotBottom','(1./GetPosCorrectionS1(' + str(self.runNumber) + ', R, Z))*ctS1sTotBottom')
				self.lT1[i].SetAlias('cpS2sTotBottom','(1./GetPosCorrectionS2(' + str(self.runNumber) + ', R, Z))*ctS2sTotBottom')

		
		self.Xrun = '(EventId != -1)' #add a cut so that add statements work
		
		self.dTOFBounds = neriX_datasets.dTOFBounds
		self.lLiqSciS1DtRange = neriX_datasets.lLiqSciS1DtRange
		
		
		
		
	def pmt_correction(self, x):
		return neriX_pmt_gain_corrections.GetGainCorrectionBottomPMT(self.runNumber, x[0])
	
	
	
	def get_run(self):
		return self.runNumber
	
	
	
	def get_filename(self, index=0):
		return self.lFilenames[index]
	
	
	
	def get_filename_no_ext(self, index=0):
		return self.lFilenames[index][:-5]
	
	
	
	def get_T1(self, index=0):
		return self.lT1[index]
	
	
	
	def get_T2(self, index=0):
		return self.lT2[index]
			
	
	
	def get_T3(self, index=0):
		return self.lT3[index]
	
	
	
	def get_T4(self, index):
		return self.lT4[index]
	
	
	
	def get_lT1(self):
		return self.lT1
	
	
	
	def get_lT2(self):
		return self.lT2
	
	
	
	def get_lT3(self):
		return self.lT3
	
	
	
	def get_lT4(self):
		return self.lT4



	def add_dt_cut(self, lowdt = 2., highdt = 13.):
		Xdt = '((dt > ' + str(lowdt) + ') && (dt < '+ str(highdt) + '))'
		self.Xrun = self.Xrun + ' && ' + Xdt



	def add_z_cut(self, lowZ = -22., highZ = -4.):
		if self.runNumber == 10 or self.runNumber == 11:
			if self.cathodeSetting == 0.345:
				Xz = '((Z > -25.1) && (Z < -3.75))'
			elif self.cathodeSetting == 1.054:
				Xz = '((Z > -25.5) && (Z < -4.1))'
			elif self.cathodeSetting == 2.356:
				Xz = '((Z > -25.8) && (Z < -4.5))'
			elif self.cathodeSetting == 5.500:
				Xz = '((Z > -26.3) && (Z < -4.9))'
		if self.runNumber == 15 or self.runNumber == 16:
			Xz = '((Z > -24.3) && (Z < -1.0))'
		else:
			Xz = '((Z > ' + str(lowZ) + ') && (Z < '+ str(highZ) + '))'
		self.Xrun = self.Xrun + ' && ' + Xz



	def add_radius_cut(self, lowRadius = None, highRadius = None):
		useDefault = False
		if lowRadius == None and highRadius == None:
			useDefault = True
		
		if lowRadius == None:
			lowRadius = 0.
		if highRadius == None:
			highRadius = 20.
	
		if useDefault and (self.runNumber == 10 or self.runNumber == 11):
			if self.cathodeSetting == 0.345:
				Xradius = '(R < 21.)'
			elif self.cathodeSetting == 1.054:
				Xradius = '(R < 21.)'
			elif self.cathodeSetting == 2.356:
				Xradius = '(R < 21.)'
			elif self.cathodeSetting == 5.500:
				Xradius = '(R < 21.)'
		else:
			Xradius = '((R > ' + str(lowRadius) + ') && (R < '+ str(highRadius) + '))'
		self.Xrun = self.Xrun + ' && ' + Xradius
		


	def add_eDep_cut(self, lowEnergy = -2., highEnergy = 35.):
		Xedep = '((eDep > '+str(lowEnergy)+') && (eDep < '+str(highEnergy)+'))'
		self.Xrun = self.Xrun + ' && ' + Xedep


	
	
	def add_temp_neutron_cut(self, angle, lEJChannels = []):
		# add temporary neutron cuts for each EJ
		Xneutron = '( (EventId == -1) ' # always false so does nothing
		
		assert isinstance(angle, int), 'Angle input should be of type int - round to nearest degree.'
		assert isinstance(self.degreeSetting, dict), 'Must be using NR coincidence data - "degreeSetting" should be a dictionary.'
		
		lChannelsToCutOn = []
		lChannelsAtAngle = self.degreeSetting[angle]
		if len(lEJChannels) > 0:
			for channel in lEJChannels:
				if not channel in lChannelsAtAngle:
					print 'Chosen EJ channel not at given angle - please fix!'
					sys.exit()
				else:
					lChannelsToCutOn.append(channel)
		else:
			lChannelsToCutOn = lChannelsAtAngle
		
		# now that channels are final, make cuts on EJs
		for channel in lChannelsToCutOn:
			if channel == 0:
				Xneutron += ' || ((psd0 > 0.22) && (LiqSciHeight[0] > 0.2 && LiqSciHeight[0] < 1.0))'
			elif channel == 1:
				Xneutron += ' || ((psd1 > 0.15) && (LiqSciHeight[1] > 0.35 && LiqSciHeight[1] < 1.0))'
			elif channel == 2:
				Xneutron += ' || ((psd2 > 0.2 && psd2 < 0.5) && (LiqSciHeight[2] > 0.4 && LiqSciHeight[2] < 1.4))'
			elif channel == 3:
				Xneutron += ' || ((psd3 > 0.24 && psd3 < 0.5) && (LiqSciHeight[3] > 0.4 && LiqSciHeight[3] < 1.35))'
				#Xneutron += ' || ((psd3 > 0.2) && (LiqSciHeight[3] > 0.4 && LiqSciHeight[3] < 1.4))'
			else:
				print 'neriX_analysis not able to handle EJ channel ' + str(channel) + '.  Please check input and try again.'
				sys.exit()
		
		Xneutron += ' )' # end if clause
		self.Xrun = self.Xrun + ' && ' + Xneutron
		
		
		
	def add_temp_gamma_cut(self, angle, lEJChannels = []):
		# add temporary neutron cuts for each EJ
		Xgamma = '( (EventId == -1) ' # always false so does nothing
		
		assert isinstance(angle, int), 'Angle input should be of type int - round to nearest degree.'
		assert isinstance(self.degreeSetting, dict), 'Must be using NR coincidence data - "degreeSetting" should be a dictionary.'
		
		lChannelsToCutOn = []
		lChannelsAtAngle = self.degreeSetting[angle]
		if len(lEJChannels) > 0:
			for channel in lEJChannels:
				if not channel in lChannelsAtAngle:
					print 'Chosen EJ channel not at given angle - please fix!'
					sys.exit()
				else:
					lChannelsToCutOn.append(channel)
		else:
			lChannelsToCutOn = lChannelsAtAngle
		
		# now that channels are final, make cuts on EJs
		for channel in lChannelsToCutOn:
			if channel == 0:
				Xgamma += ' || ((psd0 < 0.22) && (LiqSciHeight[0] > 0.2 && LiqSciHeight[0] < 1.0))'
			elif channel == 1:
				Xgamma += ' || ((psd1 < 0.15) && (LiqSciHeight[1] > 0.35 && LiqSciHeight[1] < 1.0))'
			elif channel == 2:
				Xgamma += ' || ((psd2 < 0.2) && (LiqSciHeight[2] > 0.4 && LiqSciHeight[2] < 1.4))'
			elif channel == 3:
				Xgamma += ' || ((psd3 < 0.2) && (LiqSciHeight[3] > 0.4 && LiqSciHeight[3] < 1.4))'
			else:
				print 'neriX_analysis not able to handle EJ channel ' + str(channel) + '.  Please check input and try again.'
				sys.exit()
		
		Xgamma += ' )' # end if clause
		self.Xrun = self.Xrun + ' && ' + Xgamma
		
		
	
	def add_temp_tof_cut(self, angle):
		"""
		Xtof = '( '
		if angle == 45 and self.cathodeSetting == 1.054:
			Xtof += 'TimeOfFlight > 25 && TimeOfFlight < 50'
		elif angle == 30 and self.cathodeSetting == 1.054:
			Xtof += 'TimeOfFlight > 40 && TimeOfFlight < 80'#65'
		elif angle == 45 and not self.cathodeSetting == 1.054:
			Xtof += 'TimeOfFlight > 5 && TimeOfFlight < 40'
		elif angle == 30 and not self.cathodeSetting == 1.054:
			Xtof += 'TimeOfFlight > 10 && TimeOfFlight < 50'
		else:
			print 'Currently not setup to handle ' + str(angle) + ' degrees.  Please edit neriX_analysis.py appropriately.'
			sys.exit()
		Xtof += ' )'
		"""
		Xtof = ' ( TimeOfFlight > %f && TimeOfFlight < %f ) ' % self.dTOFBounds[(angle, self.cathodeSetting)]
		self.Xrun = self.Xrun + ' && ' + Xtof



	def get_tof_window(self, angle):
		return self.dTOFBounds[(angle, self.cathodeSetting)]
	
		
		
		
	def add_s1_liqsci_peak_cut(self, lEJChannels=[0, 1, 2, 3]):
		Xpeak = '( '
		for channel in lEJChannels:
			Xpeak += '( (LiqSciPeak[%d] - S1sPeak[]) < %d && (LiqSciPeak[%d] - S1sPeak[]) > %d ) || ' % (channel, self.lLiqSciS1DtRange[1], channel, self.lLiqSciS1DtRange[0])
		Xpeak = Xpeak[:-4] + ' )'
		self.Xrun = self.Xrun + ' && ' + Xpeak
	
		
		
		
	def add_tof_cut(self, lowTime, highTime):
		Xtof = '(TimeOfFlight > %f && TimeOfFlight < %f)' % (lowTime, highTime)
		self.Xrun += ' && ' + Xtof
	
	
	
	def add_xs2asym_cut(self):
		Xs2asym = '( s2asym > -0.8*exp(-S2sTotBottom[0]/400.)-0.2 )'
		self.Xrun += ' && ' + Xs2asym
		


	def add_xs1asym_cut(self):
		Xs1asym = '( s1asym > -1.4*exp(-S1sTotBottom[0]/10.)+0.4 )'
		self.Xrun += ' && ' + Xs1asym



	def add_single_scatter_cut(self):
		if self.runNumber == 10 or self.runNumber == 11:
			self.Xrun += ' && ( (Alt$(ratio_s2tot1_s2top0,0.) < 0.06 || (((Alt$(Z,3.)>2.0 && Alt$(Z,3.)<5.0) || Alt$(Z,30.)>24.5) && (Alt$(ratio_s2tot1_s2top0,0.) < 0.1))) )'
		elif self.runNumber == 15 or self.runNumber == 16:
			self.Xrun += ' && ( Alt$(S2sTotBottom[1], -1) < 50 )'
			#self.Xrun += ' && ( Alt$(S2sTotBottom[1], -1) < 0.1*S2sTotBottom[0] )'
	
		
		
	def add_s1_trig_cut(self):
		Xtrig = ''
		if isinstance(self.degreeSetting, dict) or self.degreeSetting < 0:
			print 'Using s1 trigger cut from ER measurement incorrectly!'
			print 'Please check your code and remove cut or make appropriate corrections.'
			return
		if self.get_timestamp(0) < 1418997600: # 12/19/14 at 9 AM
			Xtrig = '((((TrigLeftEdge[] - S1sPeak[0]) > 0.) && ((TrigLeftEdge[] - S1sPeak[0]) < 50.)) && ((TrigArea[] > (2.2e-8) && TrigArea[] < (2.9e-8)) || (TrigArea[] > (8.2e-8) && TrigArea[] < (10.2e-8)) || (TrigArea[] > (1.50e-7) && TrigArea[] < (1.64e-7)) || (TrigArea[] > 2.2e-7 && TrigArea[] < 2.3e-7)))' #Xtrig3
		elif self.get_timestamp(0) > 1418997600:
			Xtrig = '((((TrigLeftEdge[] - S1sPeak[0]) > 0.) && ((TrigLeftEdge[] - S1sPeak[0]) < 50.)) && ((TrigArea[] > (3.0e-8) && TrigArea[] < (4.8e-8)) || (TrigArea[] > (9.2e-8) && TrigArea[] < (10.6e-8)) || (TrigArea[] > (1.50e-7) && TrigArea[] < (1.7e-7))))' # Xtrig4
		
		Xtrig = 'Max$(' + Xtrig + ' ? 1 : 0)'
		self.Xrun += ' && ' + Xtrig
		
		
		
	def add_cut(self, sCut):
		self.Xrun += ' && (%s)' % sCut
		
		
		
	def reset_cuts(self):
		self.Xrun = '(EventId != -1)'



	def get_degree_setting(self):
		return self.degreeSetting



	def get_cathode_setting(self):
		return self.cathodeSetting



	def get_anode_setting(self):
		return self.anodeSetting



	def get_cuts(self):
		return self.Xrun
	
	
	
	def get_num_events(self):
		numEvents = 0
		for eventList in self.lEList:
			numEvents += eventList.GetN()
		return numEvents
	
	
	
	def get_num_events_before_cuts(self):
		numEvents = 0
		for tree in self.lT1:
			numEvents += tree.GetEntriesFast()
		return numEvents



	def get_timestamp(self, eventNumber=0, index=0):
		self.lT1[index].GetEntry(eventNumber)
		return self.lT1[index].TimeSec



	def get_livetime(self):
		# use -1 to grab last file in list
		totalTime = 0.
		for i in xrange(len(self.lT1)):
			totalTime += (self.get_timestamp(self.lT1[i].GetEntries() - 1, i) - self.get_timestamp(0, i))
		return totalTime



	def set_event_list(self, cuts = None):
		if cuts == None:
			cuts = self.Xrun
		#self.reset_event_list()
		for i, currentTree in enumerate(self.lT1):
			print 'Working on ' + str(self.lFilenames[i]) + ' (' + str(i+1) + '/' + str(len(self.lT1)) + ')...'
			print 'Original number of elements: ' + str(currentTree.GetEntries())
			root.TTree.Draw(currentTree, '>>eList_' + self.lFilenames[i], root.TCut(cuts), '')
			print 'Number of elements after cuts: ' + str(self.lEList[i].GetN())
			self.lT1[i].SetEventList(self.lEList[i])



	def thread_set_event_list(self, lIndices, lock, cuts = None):
		if cuts == None:
			cuts = self.Xrun
		#self.reset_event_list()
		for i in lIndices:
			with lock:
				print 'Working on ' + str(self.lFilenames[i]) + ' (' + str(i+1) + '/' + str(len(self.lT1)) + ')...'
				print 'Original number of elements: ' + str(self.lT1[i].GetEntries()) + ' (' + str(i+1) + '/' + str(len(self.lT1)) + ')'
			root.TTree.Draw(self.lT1[i], '>>eList_' + self.lFilenames[i], root.TCut(cuts), '')
			with lock:
				print 'Number of elements after cuts: ' + str(self.lEList[i].GetN()) + ' (' + str(i+1) + '/' + str(len(self.lT1)) + ')'
			self.lT1[i].SetEventList(self.lEList[i])



	def multithread_set_event_list(self, numThreads=1, cuts=None):
		if numThreads == 1:
			self.set_event_list(cuts)
		else:
			if numThreads > len(self.lT1):
				print 'More threads than files - reducing number of processors used'
				numThreads = len(self.lT1)
			# set thread tasks
			lThreadTasks = [[] for i in xrange(numThreads)]
			for i in xrange(len(self.lFilenames)):
				lThreadTasks[i%numThreads].append(i)

			# call worker function
			lThreads = [0. for i in xrange(numThreads)]
			lock = threading.Lock()
			for i in xrange(numThreads):
				lThreads[i] = threading.Thread(target=self.thread_set_event_list, args=(lThreadTasks[i], lock, cuts))
				lThreads[i].start()

			# block calling process until threads finish
			for i in xrange(numThreads):
				lThreads[i].join()





	# overwrite standard draw to accomodate list of files
	def Draw(self, *args, **kwargs):
		for currentTree in self.lT1:
			currentTree.Draw(*args, **kwargs)



	def draw(self, *args, **kwargs):
		self.Draw(*args, **kwargs)
	
	
	
	def thread_draw(self, lock, lTreeIndeces, args, kwargs):
		for index in lTreeIndeces:
			with lock:
				print 'Currently drawing ' + str(self.lFilenames[index]) + ' to histogram'  + ' (' + str(index+1) + '/' + str(len(self.lT1)) + ')...'
			self.lT1[index].Draw(*args, **kwargs)


	
	# multithreads the draw command
	# call draw as you normally would BUT with additional two arguments at the beginning
	def multithread_draw(self, numThreads=1, *origArgs, **kwargs):
		try:
			origHist = kwargs['hist']
		except KeyError:
			print 'Must pass "hist" argument to draw'
			sys.exit()
		if numThreads == 1:
			self.Draw(*origArgs, **kwargs)
		else:
			if numThreads > len(self.lT1):
				print 'More threads than files - reducing number of processors used'
				numThreads = len(self.lT1)
			lock = threading.Lock()
			
			lHists = [0 for i in xrange(numThreads)]
			lKwargs = [0 for i in xrange(numThreads)]
			lThreads = [0 for i in xrange(numThreads)]
			
			# set thread tasks
			lThreadTasks = [[] for i in xrange(numThreads)]
			for i in xrange(len(self.lFilenames)):
				lThreadTasks[i%numThreads].append(i)
			
			for i in xrange(numThreads):
				lHists[i] = origHist.empty_clone()
				lKwargs[i] = dict(kwargs)
				try:
					lKwargs[i]['hist'] = lHists[i]
				except:
					print 'Must pass "hist" argument to draw'
					sys.exit()
				
				lThreads[i] = threading.Thread(target=self.thread_draw, args=(lock, lThreadTasks[i], origArgs, lKwargs[i]))
				lThreads[i].start()

			# block calling process until threads finish
			for i in xrange(numThreads):
				lThreads[i].join()

			for i in xrange(numThreads):
				origHist.Add(lHists[i])


	
	
	def reset_event_list(self):
		self.T1.SetEventList(0)
		if ('elist_' + self.filename) in locals():
			self.eList.Clear()
			self.eList.Reset()
			self.eList.SetDirectory(0)
			#del self.eList



	def get_event_list(self, index=0):
		return self.lEList[index]












#test = neriX_analysis('nerix_140914_1631.root')
#test.get_timestamp(1)
#print test.get_livetime()