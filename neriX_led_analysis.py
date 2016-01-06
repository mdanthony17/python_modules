import ROOT as root
from ROOT import gROOT
import sys, os.path, os
from rootpy import stl
from rootpy.io import File
from rootpy.tree import Tree, TreeModel, TreeChain
from rootpy.plotting import Hist, Canvas
import neriX_config, neriX_led_datasets, neriX_pmt_gain_corrections
import threading, pickle, time, array

# MUST INCLUDE TYPES TO AVOID SEG FAULT
stl.vector(stl.vector('float'))
stl.vector(stl.vector('int'))


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


# neriX_led_analysis
	# get_run(self): returns the run number
	# fit_pmt_channel(self, channelNumber, bypassApproval = False): automates the fit of a single PMT channel
	# manual_fit(self, channelNumber): gives the user a chance to manually fit the spe spectrum if the automatic fit is failing
	# change_parameter_value_subroutine(self, dFitParametersToChange): subroutine for fitting functions - do not call outside of member function
	# add_spe_results_to_file(self): pickles the fit results for later update to mysql
	# *fit_all_channels(self, bypassApproval=False): the workhorse function - attempts to automate all channels and then brings up manual mode for ones that are not approved or where the fits fail
	# plot_gain_vs_time(self, channelNumber): plots gain graph vs time from pickle file
	# *plot_all_channels_gain_vs_time(self): plots gain graphs for all channels







class neriX_led_analysis:
	def __init__(self, sNameOfSingleFile, forceReprocess = False, timeCut = True):
		# features to add
		# 1. Optimize gain fitting procedure
		# 2. Print out canvas with signal, bkg, and bkg subtraction hist
		# 3. Update pickle file to mysql
	
	
		self.lLightLevels = neriX_led_datasets.lLightLevels
		dRunFiles = neriX_led_datasets.run_files
		lRuns = neriX_led_datasets.runsInUse
		self.lHistParameters = neriX_led_datasets.lHistParameters
		
		sPathToData = neriX_led_datasets.sPathToData # must append run number
		sPathToProcessedData = neriX_led_datasets.sPathToProcessedData
		
		self.dLightLevels = neriX_led_datasets.dLightLevels
		self.stringForNumEntries = 'num_events'
		
		self.sNameOfSingleFile = sNameOfSingleFile
		
		# append .root if not there
		if sNameOfSingleFile[-5:] != '.root':
			sNameOfSingleFile += '.root'
		
		dChannelsAtLightLevels = neriX_led_datasets.dChannelsAtLightLevels
		self.lChannelsToAnalyze = []

		self.dFiles = None
		for run in lRuns:
			for dCheckFiles in dRunFiles[run]:
				for lightLevel in dCheckFiles:
					if sNameOfSingleFile == dCheckFiles[lightLevel]:
						currentRun = run
						self.dFiles = dCheckFiles
					else:
						continue
			
		if not self.dFiles:
			print str(sNameOfSingleFile) + ' does not appear in neriX_led_datasets.py'
			print 'Please check if the file exists and that your spelling is correct'
			sys.exit()
		
		
		self.runNumber = currentRun
		self.sPathToGraphs = neriX_led_datasets.sPathToGraphs + 'run_' + str(self.runNumber) + '/'
		self.sPathToHists = neriX_led_datasets.sPathToHists + 'run_' + str(self.runNumber) + '/'
		sPathToData += 'run_' + str(self.runNumber) + '/'
		lProcessedPreviously = [False for i in self.lLightLevels]
		
		self.lProcessedRootFiles = []
			
		for i, lightLevel in enumerate(self.dFiles):
			# see if data has been processed already
			if os.path.isfile(sPathToProcessedData + 'spe_' + self.dFiles[lightLevel]) and not forceReprocess:
				print 'Found processed data for ' + self.dFiles[lightLevel]
				self.lProcessedRootFiles.append(File(sPathToProcessedData + 'spe_' + self.dFiles[lightLevel], 'read'))
				
			else:
				print 'Processed data not found for ' + str(self.dFiles[lightLevel]) + ', processing now...'
				if self.dFiles[lightLevel][-5:] != '.root':
					self.dFiles[lightLevel] += '.root'

				# check if file exists on computer
				if not os.path.isfile(sPathToData + self.dFiles[lightLevel]):
					print 'Could not find file ' + str(self.dFiles[lightLevel]) + ' in ' + str(sPathToData)
					print 'Exiting...'
					sys.exit()
					
				# load root file
				try:
					currentFile = File(sPathToData + self.dFiles[lightLevel], 'r')
				except:
					print 'Could not load ' + str(currentFile) + ' - please check quality of file.'
					sys.exit()
		
				T0 = currentFile.T0
				T0.SetName('T0_' + self.dFiles[lightLevel])
				T0.create_buffer()
				
				# all files must have same number of entries
				self.originalNumberOfEvents = T0.GetEntriesFast()
				
				processedFile = File(sPathToProcessedData + 'spe_' + self.dFiles[lightLevel], 'recreate')
				
				lHists = [Hist(self.lHistParameters[0], self.lHistParameters[1], self.lHistParameters[2], name='spe_' + str(i+1), title='spe_' + str(i+1), drawstyle='hist') for i in xrange(17)]
				for i in xrange(17):
					xTime = ''
					if timeCut:
						xTime = 'SingleSample['+str(i)+'] > 157 && SingleSample['+str(i)+'] < 188'
					T0.Draw('SingleBefore[%d][1]+SingleBefore[%d][0]+SinglePeak[%d]+SingleAfter[%d][0]+SingleAfter[%d][1]' % (5*(i,)), hist=lHists[i], selection=xTime)
					lHists[i].Sumw2()
					processedFile.cd()
					lHists[i].Write(lHists[i].GetName())
					
				sNumEntries = root.TObjString(str(self.originalNumberOfEvents))
				sNumEntries.Write(self.stringForNumEntries, root.TObject.kOverwrite)
					
				self.lProcessedRootFiles.append(processedFile)
			self.lChannelsToAnalyze += dChannelsAtLightLevels[lightLevel]
					
		# fit approval by user:
		# -1: fit failed, 0: not completed yet, 1: passed!
		self.dFitApproved = {i:0 for i in self.lChannelsToAnalyze}
	
		# final parameters from fits
		self.dFitParameters = {i:{'bkg_mean':-1, 'bkg_width':-1, 'spe_mean':-1, 'spe_width':-1, 'bkg_mean_err':-1, 'bkg_width_err':-1, 'spe_mean_err':-1, 'spe_width_err':-1, 'chi2':-1, 'ndf':-1} for i in self.lChannelsToAnalyze}
		
		# check if pickle file exists
		# try to load pickle file that has saved histogram parameters
		try:
			self.dCompletedFitParameters = pickle.load(open('completed_spe_fits.p', 'r'))
		except:
			print 'Pickled file of old parameters does not exist yet'
			self.dCompletedFitParameters = {}

		self.numEventsBeforeCuts = int(str(self.lProcessedRootFiles[0].Get(self.stringForNumEntries)))
		

	
	def get_run(self):
		return self.runNumber



	def fit_pmt_channel(self, channelNumber, bypassApproval = False):
		print '\n\n---------- Automated fit of channel ' + str(channelNumber) + ' ----------\n\n'
	
		assert channelNumber >= 1 and channelNumber <= 17 and type(channelNumber) == type(1), 'Improper channel number given - please check input'
		
		failStatus = -1
	
		# grab required histograms
		bkgFile = self.lProcessedRootFiles[0]
		signalFile = self.lProcessedRootFiles[self.dLightLevels[channelNumber]]

		hBkg = bkgFile.Get('spe_' + str(channelNumber))
		hSignal = signalFile.Get('spe_' + str(channelNumber))
		hSignal.DrawStyle = 'hist'
		
		lowBkgTrial = -1e5
		highBkgTrial = 8e5
		
		lowBkgSubtractTrial = 1e6
		highBkgSubtractTrial = 3.5e6
		
		# used in setting limits of actual fits after trials
		trialWidthFactor = 2
		individualFitSensitivity = 0.6




		# step 1: find approximate location of peak via bkg subtraction
		fBkgGausTrial = root.TF1('fBkgGausTrial', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		fitResult = hBkg.Fit(fBkgGausTrial, 'NQMELS+', '', lowBkgTrial, highBkgTrial)
		
		if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
			print 'Fit of bkg trial failed!!!'
			return failStatus

		bkgTrialMean = fBkgGausTrial.GetParameter(1)
		bkgTrialWidth = fBkgGausTrial.GetParameter(2)

		fBkgGaus = root.TF1('fBkgGaus', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		fitResult = hBkg.Fit(fBkgGaus, 'NQSMEL+', '', bkgTrialMean - trialWidthFactor*bkgTrialWidth, bkgTrialMean + trialWidthFactor*bkgTrialWidth)

		if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
			print 'Fit of bkg failed!!!'
			return failStatus

		bkgMean = fBkgGaus.GetParameter(1)
		bkgWidth = fBkgGaus.GetParameter(2)
		bkgMeanErr = fBkgGaus.GetParError(1)
		bkgWidthErr = fBkgGaus.GetParError(2)



		# step 2: find approximate location of bkg peak in bkg file
		hBkgSubtract = hSignal.Clone('hBkgSubtract')
		hBkgSubtract.Add(hBkg, -1.)

		fBkgSubtractGausTrial = root.TF1('fBkgSubtractGausTrial', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		fitResult = hBkgSubtract.Fit(fBkgSubtractGausTrial, 'NQSMEL+', '', lowBkgSubtractTrial, highBkgSubtractTrial)
		
		if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
			print 'Fit of bkg subtract trial failed!!!'
			return failStatus

		bkgSubtractTrialMean = fBkgSubtractGausTrial.GetParameter(1)
		bkgSubtractTrialWidth = fBkgSubtractGausTrial.GetParameter(2)
		lowEndFit = bkgSubtractTrialMean - trialWidthFactor*bkgSubtractTrialWidth
		highEndFit = bkgSubtractTrialMean + trialWidthFactor*bkgSubtractTrialWidth

		fBkgSubtractGaus = root.TF1('fBkgSubtractGaus', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		fitResult = hBkgSubtract.Fit(fBkgSubtractGaus, 'NSMEL+', '', lowEndFit, highEndFit)

		if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
			print 'Fit of bkg subtract failed!!!'
			return failStatus

		# check bkg subtracted fit
		#hBkgSubtract.drawstyle = 'hist'
		#hBkgSubtract.Sumw2()
		hBkgSubtract.SetStats(0)
		c1 = Canvas()

		bkgSubtractMean = fBkgSubtractGaus.GetParameter(1)
		bkgSubtractWidth = fBkgSubtractGaus.GetParameter(2)
		bkgSubtractMeanErr = fBkgSubtractGaus.GetParError(1)
		bkgSubtractWidthErr = fBkgSubtractGaus.GetParError(2)
		bkgSubtractChi2 = fBkgSubtractGaus.GetChisquare()
		bkgSubtractNDF = fBkgSubtractGaus.GetNDF()

		hBkgSubtract.Draw('E1')
		hBkgSubtract.GetXaxis().SetRangeUser(0, 1.1*highEndFit)
		fBkgSubtractGaus.Draw('same')
		
		meanNumPE = hBkgSubtract.Integral(hBkgSubtract.FindBin(0), hBkgSubtract.FindBin(self.lHistParameters[2])) / self.numEventsBeforeCuts
		
		fitInfo = 'Mean = %.3e +/- %.2e, #mu_{spe} = %.3f' % (bkgSubtractMean, bkgSubtractMeanErr, meanNumPE)
		pt1 = root.TPaveText(.5, .85, .9, .9, 'blNDC"')
		text1 = pt1.AddText(fitInfo)
		pt1.SetTextColor(root.kAzure + 1)
		pt1.SetFillStyle(0)
		pt1.SetBorderSize(0)
		pt1.Draw('SAME')
		
		#c1.SetLogy()
		c1.Update()

		if not bypassApproval:
			print 'Please examine fit and if you approve please enter "y" otherwise press any other key.'
			approvalResponse = raw_input('Please enter response: ')
		
		if bypassApproval or approvalResponse == 'y':
			self.dFitParameters[channelNumber]['bkg_mean'] = bkgMean
			self.dFitParameters[channelNumber]['bkg_width'] = bkgWidth
			self.dFitParameters[channelNumber]['spe_mean'] = bkgSubtractMean
			self.dFitParameters[channelNumber]['spe_width'] = bkgSubtractWidth
			self.dFitParameters[channelNumber]['bkg_mean_err'] = bkgMeanErr
			self.dFitParameters[channelNumber]['bkg_width_err'] = bkgWidthErr
			self.dFitParameters[channelNumber]['spe_mean_err'] = bkgSubtractMeanErr
			self.dFitParameters[channelNumber]['spe_width_err'] = bkgSubtractWidthErr
			self.dFitParameters[channelNumber]['chi2'] = bkgSubtractChi2
			self.dFitParameters[channelNumber]['ndf'] = bkgSubtractNDF
			self.dFitParameters[channelNumber]['mean_pe'] = meanNumPE
			
			if not os.path.isdir(self.sPathToHists + self.sNameOfSingleFile):
				os.mkdir(self.sPathToHists + self.sNameOfSingleFile)
			
			c1.SaveAs(self.sPathToHists + self.sNameOfSingleFile + '/gain_pmt_' + str(channelNumber) + '.png')
			
			return 1
		else:
			return failStatus



		# ---------- step 3 (optional and unfinished - needs optimization)
		"""
		lowEndFit = bkgMean-2*bkgWidth
		highEndFit = bkgSubtractMean+1.5*bkgSubtractWidth

		# step 3: fit with two gaussian in normal file
		fFullFit = root.TF1('fFullFit', '[0] * ( exp(-0.5 * ((x-[1])/[2])^2) + [3]*exp(-0.5 * ((x-[4])/[5])^2) )', lowEndFit, highEndFit)
		fFullFit.SetParameters(1e3, bkgMean, bkgWidth, 0.1, bkgSubtractMean, bkgSubtractWidth)
		
		fFullFit.SetParLimits(0, 200, 2e4)
		fFullFit.SetParLimits(1, bkgMean*(1-individualFitSensitivity), bkgMean*(1+individualFitSensitivity))
		fFullFit.SetParLimits(2, bkgWidth*(1-individualFitSensitivity), bkgWidth*(1+individualFitSensitivity))
		fFullFit.SetParLimits(3, 0, 2)
		fFullFit.SetParLimits(4, bkgSubtractMean*(1-individualFitSensitivity), bkgSubtractMean*(1+individualFitSensitivity))
		fFullFit.SetParLimits(5, bkgSubtractWidth*(1-individualFitSensitivity), bkgSubtractWidth*(1+individualFitSensitivity))


		hSignal.drawstyle = 'hist'
		hSignal.Sumw2()
		c1 = Canvas()

		hSignal.Draw('E1')
		hSignal.GetXaxis().SetRangeUser(0.9*lowEndFit, 1.1*highEndFit)
		fitResult = hSignal.Fit(fFullFit, 'NSL+', '', bkgMean - trialWidthFactor*bkgWidth, bkgSubtractMean + trialWidthFactor*bkgSubtractWidth)
		fFullFit.Draw('same')
		c1.SetLogy()
		c1.Update()
		
		print 'Please examine fit and if you approve please enter "y" otherwise press any other key.'
		approvalResponse = raw_input('Please enter response: ')
		
		if approvalResponse == 'y':
			return 1
		else:
			return failStatus
		"""
		# ---------- end step 3
		
		
		
	def manual_fit(self, channelNumber):
		print '\n\n---------- Manual fit of channel ' + str(channelNumber) + ' ----------\n\n'
	
		assert channelNumber >= 1 and channelNumber <= 17 and type(channelNumber) == type(1), 'Improper channel number given - please check input'
		
		failStatus = -1
	
		# grab required histograms
		bkgFile = self.lProcessedRootFiles[0]
		signalFile = self.lProcessedRootFiles[self.dLightLevels[channelNumber]]

		hBkg = bkgFile.Get('spe_' + str(channelNumber))
		hSignal = signalFile.Get('spe_' + str(channelNumber))
		hSignal.DrawStyle = 'hist'
		
		lowBkgTrial = -10e5
		highBkgTrial = 10e5
		
		lowBkgSubtractTrial = 1e6
		highBkgSubtractTrial = 3.5e6
		
		# used in setting limits of actual fits after trials
		trialWidthFactor = 1
		individualFitSensitivity = 0.6
	
	
		# repeat procedure from before
		# background should never fail
		fBkgGausTrial = root.TF1('fBkgGausTrial', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		fitResult = hBkg.Fit(fBkgGausTrial, 'NQMELS+', '', lowBkgTrial, highBkgTrial)
		
		if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
			print 'Fit of bkg trial failed!!!'
			return failStatus

		bkgTrialMean = fBkgGausTrial.GetParameter(1)
		bkgTrialWidth = fBkgGausTrial.GetParameter(2)

		fBkgGaus = root.TF1('fBkgGaus', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		fitResult = hBkg.Fit(fBkgGaus, 'NSQMEL+', '', bkgTrialMean - trialWidthFactor*bkgTrialWidth, bkgTrialMean + trialWidthFactor*bkgTrialWidth)

		if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
			print 'Fit of bkg failed!!!'
			return failStatus

		bkgMean = fBkgGaus.GetParameter(1)
		bkgWidth = fBkgGaus.GetParameter(2)
		bkgMeanErr = fBkgGaus.GetParError(1)
		bkgWidthErr = fBkgGaus.GetParError(2)
	
	
	
		# move to bkg subtract default
		hBkgSubtract = hSignal.Clone('hBkgSubtract')
		hBkgSubtract.Add(hBkg, -1.)

		fBkgSubtractGausTrial = root.TF1('fBkgSubtractGausTrial', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		fitResult = hBkgSubtract.Fit(fBkgSubtractGausTrial, 'NQSMEL+', '', lowBkgSubtractTrial, highBkgSubtractTrial)
		
		if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
			print 'Fit of bkg subtract trial failed!!!'

		bkgSubtractTrialMean = fBkgSubtractGausTrial.GetParameter(1)
		bkgSubtractTrialWidth = fBkgSubtractGausTrial.GetParameter(2)
		lowEndFit = bkgSubtractTrialMean - trialWidthFactor*bkgSubtractTrialWidth
		highEndFit = bkgSubtractTrialMean + trialWidthFactor*bkgSubtractTrialWidth

		fBkgSubtractGaus = root.TF1('fBkgSubtractGaus', 'gaus', self.lHistParameters[1], self.lHistParameters[2])
		
		c1 = Canvas()
		dFitParametersToChange = {'lower_limit_fit':lowEndFit, 'upper_limit_fit':highEndFit}

		while 1:
			fitResult = hBkgSubtract.Fit(fBkgSubtractGaus, 'NSMEL+', '', float(dFitParametersToChange['lower_limit_fit']), float(dFitParametersToChange['upper_limit_fit']))
			hBkgSubtract.SetStats(0)
			hBkgSubtract.Draw('E1')
			hBkgSubtract.GetXaxis().SetRangeUser(0, 1.1*highEndFit)
			fBkgSubtractGaus.Draw('same')
			
			
			bkgSubtractMean = fBkgSubtractGaus.GetParameter(1)
			bkgSubtractWidth = fBkgSubtractGaus.GetParameter(2)
			bkgSubtractMeanErr = fBkgSubtractGaus.GetParError(1)
			bkgSubtractWidthErr = fBkgSubtractGaus.GetParError(2)
			bkgSubtractChi2 = fBkgSubtractGaus.GetChisquare()
			bkgSubtractNDF = fBkgSubtractGaus.GetNDF()
			
			meanNumPE = hBkgSubtract.Integral(hBkgSubtract.FindBin(0), hBkgSubtract.FindBin(self.lHistParameters[2])) / self.numEventsBeforeCuts
			
			fitInfo = 'Mean = %.3e +/- %.2e, #mu_{spe} = %.3f' % (bkgSubtractMean, bkgSubtractMeanErr, meanNumPE)
			pt1 = root.TPaveText(.5, .85, .9, .9, 'blNDC"')
			text1 = pt1.AddText(fitInfo)
			pt1.SetTextColor(root.kAzure + 1)
			pt1.SetFillStyle(0)
			pt1.SetBorderSize(0)
			pt1.Draw('SAME')
			
			#c1.SetLogy()
			c1.Update()

			acceptFit = False
			if fitResult.Status() % 10 != 0 or fitResult.Status() % 100 != 0 or fitResult.Status() % 1000 != 0:
				print 'Fit failed so must try new values or quit.'
				dFitParametersToChange = self.change_parameter_value_subroutine(dFitParametersToChange)
			else:
				print 'Would you like to keep this fit, quit out (if spectrum is unlikely to ever give converging fit), or try to fit with new values?'
				print 'Type "y" to accept the fit.'
				print 'Type "q" to quit this channel\'s manual fit'
				print 'Press enter to try new parameters'
				response = raw_input('Please enter response: ')
				
				if response == 'y':
					self.dFitParameters[channelNumber]['bkg_mean'] = bkgMean
					self.dFitParameters[channelNumber]['bkg_width'] = bkgWidth
					self.dFitParameters[channelNumber]['spe_mean'] = bkgSubtractMean
					self.dFitParameters[channelNumber]['spe_width'] = bkgSubtractWidth
					self.dFitParameters[channelNumber]['bkg_mean_err'] = bkgMeanErr
					self.dFitParameters[channelNumber]['bkg_width_err'] = bkgWidthErr
					self.dFitParameters[channelNumber]['spe_mean_err'] = bkgSubtractMeanErr
					self.dFitParameters[channelNumber]['spe_width_err'] = bkgSubtractWidthErr
					self.dFitParameters[channelNumber]['chi2'] = bkgSubtractChi2
					self.dFitParameters[channelNumber]['ndf'] = bkgSubtractNDF
					self.dFitParameters[channelNumber]['mean_pe'] = meanNumPE
			
					if not os.path.isdir(self.sPathToHists + self.sNameOfSingleFile):
						os.mkdir(self.sPathToHists + self.sNameOfSingleFile)
					
					c1.SaveAs(self.sPathToHists + self.sNameOfSingleFile + '/gain_pmt_' + str(channelNumber) + '.png')
					
					return 1
				elif response == 'q':
					del self.dFitParameters[channelNumber]
					return failStatus
				else:
					dFitParametersToChange = self.change_parameter_value_subroutine(dFitParametersToChange)




	def change_parameter_value_subroutine(self, dFitParametersToChange):
		print 'You can change the following parameters:'
		print dFitParametersToChange.keys()
		while 1:
			# if adding more parameters need to generalize
			response = ''
			while response != 'l' and response != 'u' and response != 'c':
				print 'Please enter "l" to change lower limit, "u" to change upper limit, and "c" when you are satisfied with your choices or "q" to quit'
				response = raw_input('Please enter choice: ')
			if response == 'l':
				value = raw_input('Choose value for lower limit: ')
				dFitParametersToChange['lower_limit_fit'] = value
			elif response == 'u':
				value = raw_input('Choose value for upper limit: ')
				dFitParametersToChange['upper_limit_fit'] = value
			elif response == 'q':
				return dFitParametersToChange
			else:
				return dFitParametersToChange
	
	
	
	def add_spe_results_to_file(self):
		print '\n\nWARNING: this will overwrite old results from the same dataset.'
		print 'If you would like to proceed please enter "y" or press enter to exit'
		response = raw_input('Please enter response: ')
		
		if response == 'y':
			if not (self.runNumber in self.dCompletedFitParameters):
				self.dCompletedFitParameters[self.runNumber] = {}
			
			self.dCompletedFitParameters[self.runNumber][convert_name_to_unix_time(self.sNameOfSingleFile)] = self.dFitParameters
			pickle.dump(self.dCompletedFitParameters, open('completed_spe_fits.p', 'w'))
	
	
	
	def remove_spe_results_from_file(self):
		print '\n\nWARNING: this will delete old results from the system.'
		print 'If you would like to proceed please enter "y" or press enter to exit'
		response = raw_input('Please enter response: ')
		
		if response == 'y':
			if not (self.runNumber in self.dCompletedFitParameters):
				self.dCompletedFitParameters[self.runNumber] = {}
			
			del self.dCompletedFitParameters[self.runNumber][convert_name_to_unix_time(self.sNameOfSingleFile)]
			pickle.dump(self.dCompletedFitParameters, open('completed_spe_fits.p', 'w'))




	def fit_all_channels(self, bypassApproval=False):
		lFailedFitChannels = []
		for channelNumber in self.lChannelsToAnalyze:
			self.dFitApproved[channelNumber] = self.fit_pmt_channel(channelNumber, bypassApproval)
			if self.dFitApproved[channelNumber] == -1:
				lFailedFitChannels.append(channelNumber)

		print 'The following fits failed or were not approved:'
		print lFailedFitChannels
		print 'Proceeding to manual fitting...'
		for channelNumber in lFailedFitChannels:
			self.manual_fit(channelNumber)

		self.add_spe_results_to_file()




	def plot_gain_vs_time(self, channelNumber):
		# take graph parameters from led_datasets
		lGraphRangeY = neriX_led_datasets.lGraphRangeY
	
		# grab list of parameters from pickle dictionary
		# and sort them for graph
		aUnsortedTimes = array.array('d')
		aUnsortedTimesErr = array.array('d')
		aUnsortedGains = array.array('d')
		aUnsortedGainsErr = array.array('d')
	
		aTimes = array.array('d')
		aTimesErr = array.array('d')
		aGains = array.array('d')
		aGainsErr = array.array('d')
		
		numElements = 0
		
		for time in self.dCompletedFitParameters[self.runNumber]:
			try:
				aUnsortedGains.append(self.dCompletedFitParameters[self.runNumber][time][channelNumber]['spe_mean'])
				aUnsortedGainsErr.append(self.dCompletedFitParameters[self.runNumber][time][channelNumber]['spe_mean_err'])
			except KeyError:
				continue
			aUnsortedTimes.append(time)
			aUnsortedTimesErr.append(0)
			numElements += 1
		
		if len(aUnsortedTimes) == 0:
			return 0

		points = zip(aUnsortedTimes, aUnsortedGains, aUnsortedTimesErr, aUnsortedGainsErr)
		sorted_points = sorted(points)
		
		for point in sorted_points:
			aTimes.append(point[0])
			aGains.append(point[1])
			aTimesErr.append(point[2])
			aGainsErr.append(point[3])


		# now that points are sorted create graph
		c1 = Canvas()
		gGainVsTime = root.TGraphErrors(numElements, aTimes, aGains, aTimesErr, aGainsErr)
		gGainVsTime.SetMarkerColor(root.kBlue)
		gGainVsTime.SetLineColor(root.kBlue)
		gGainVsTime.SetTitle('Channel ' + str(channelNumber) + ' - Gain run_' + str(self.runNumber))
		gGainVsTime.GetXaxis().SetTimeDisplay(1);
		gGainVsTime.GetXaxis().SetTimeFormat('%m-%d')
		gGainVsTime.GetXaxis().SetTimeOffset(0, 'gmt')
		gGainVsTime.GetYaxis().SetRangeUser(lGraphRangeY[0], lGraphRangeY[1])
		gGainVsTime.GetYaxis().SetTitle('SPE gain [electrons]')
		gGainVsTime.Draw('AP')
		
		c1.SaveAs(self.sPathToGraphs + '/gain_pmt_' + str(channelNumber) + '.png')
		c1.SaveAs(self.sPathToGraphs + '/gain_pmt_' + str(channelNumber) + '.C')
	
	



	def plot_all_channels_gain_vs_time(self):
		for channelNumber in xrange(1, 18):
			self.plot_gain_vs_time(channelNumber)



	def draw_channel(self, channelNumber):
		print '\n\n---------- Draw channel ' + str(channelNumber) + ' ----------\n\n'
	
		assert channelNumber >= 1 and channelNumber <= 17 and type(channelNumber) == type(1), 'Improper channel number given - please check input'
		
		failStatus = -1
	
		# grab required histograms
		bkgFile = self.lProcessedRootFiles[0]
		signalFile = self.lProcessedRootFiles[self.dLightLevels[channelNumber]]

		hSignal = signalFile.Get('spe_' + str(channelNumber))
		hSignal.DrawStyle = 'hist'

		c1 = Canvas()
		hSignal.Draw('')
		c1.Update()

		raw_input('Please press enter when finished...')





if __name__ == '__main__':
	test = neriX_led_analysis('nerix_150819_1511')
	test.fit_all_channels()
	test.plot_all_channels_gain_vs_time()