import ROOT as root
from ROOT import gROOT
import sys
from rootpy import stl
from rootpy.io import File
from rootpy.tree import Tree, TreeModel, TreeChain
import neriX_config, neriX_datasets, neriX_pmt_gain_corrections

# MUST INCLUDE TYPES TO AVOID SEG FAULT
stl.vector(stl.vector('float'))
stl.vector(stl.vector('int'))

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
			
		if numFiles == 1:
			self.anodeSetting = lParameters[ANODE_INDEX]
			self.cathodeSetting = lParameters[CATHODE_INDEX]
			self.degreeSetting = lParameters[DEGREE_INDEX]
		else:
			self.anodeSetting = anodeSetting
			self.cathodeSetting = cathodeSetting
			self.degreeSetting = degreeSetting
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
						
					self.lRootFiles.append(File(pathToFile, 'read'))
					if self.lRootFiles[-1].keys() == []:
						print 'Problem opening file - please check name entered'
						print 'Entered: ' + pathToFile
						self.lRootFiles.pop()
					print 'Successfully added ' + str(file[-22:]) + '!\n'
					self.lFilenames.append(file)
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
			self.lT1[i].SetAlias('X','S2sPosFann[0][0]')
			self.lT1[i].SetAlias('Y','S2sPosFann[0][1]')
			if self.lT4[i]:
				self.lT1[i].SetAlias('R','sqrt(pow(ctNNPos[0],2)+pow(ctNNPos[1],2))')
			else:
				self.lT1[i].SetAlias('R','sqrt(pow(S2sPosFann[0][0],2)+pow(S2sPosFann[0][1],2))')
			self.lT1[i].SetAlias('czS1sTotBottom','(1./(0.863098 + (-0.00977873)*Z))*S1sTotBottom')
			
			# need to create compiled function for ct alias
			gROOT.ProcessLine('.L ' + str(neriX_config.pathToModules) + 'neriX_gain_correction.C+')
			self.lT1[i].SetAlias('ctS1sTotBottom','(GetGainCorrectionBottomPMT(' + str(self.runNumber) + ', TimeSec))*czS1sTotBottom')
			
			
			self.lT1[i].SetAlias('s1asym', '(S1sTotTop[0]-S1sTotBottom[0])/S1sTot[0]')
			self.lT1[i].SetAlias('s2asym', '(S2sTotTop[0]-S2sTotBottom[0])/S2sTot[0]')
			
			if self.cathodeSetting == 0.345:
				self.lT1[i].SetAlias('Z','-1.511*(S2sPeak[0]-S1sPeak[0])/100')
			elif self.cathodeSetting == 1.054:
				self.lT1[i].SetAlias('Z','-1.717*(S2sPeak[0]-S1sPeak[0])/100')
			elif self.cathodeSetting == 2.356:
				self.lT1[i].SetAlias('Z','-1.958*(S2sPeak[0]-S1sPeak[0])/100')
			elif self.cathodeSetting == 5.500:
				self.lT1[i].SetAlias('Z','-2.208*(S2sPeak[0]-S1sPeak[0])/100')
			else:
				print 'Incorrect field entered - cannot correct Z'

			#will need to change these constants depending on Ge Calibratio
			self.lT1[i].SetAlias('GeEnergy',self.__GERMANIUM_SLOPE + '*GeHeight[0] + ' + self.__GERMANIUM_INTERCEPT)
			self.lT1[i].SetAlias('eDep','661.657 - GeEnergy')

		
		self.Xrun = '(EventId != -1)' #add a cut so that add statements work
		
		
		
	def pmt_correction(self, x):
		return neriX_pmt_gain_corrections.GetGainCorrectionBottomPMT(self.runNumber, x[0])
	
	
	
	def get_run(self):
		return self.runNumber
	
	
	
	def get_filename(self, index=0):
		return self.lFilenames[index]
	
	
	
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



	def add_radius_cut(self, lowRadius = 0., highRadius = 20.):
		Xradius = '((R > ' + str(lowRadius) + ') && (R < '+ str(highRadius) + '))'
		self.Xrun = self.Xrun + ' && ' + Xradius
		


	def add_eDep_cut(self, lowEnergy = -2., highEnergy = 35.):
		Xedep = '((eDep > '+str(lowEnergy)+') && (eDep < '+str(highEnergy)+'))'
		self.Xrun = self.Xrun + ' && ' + Xedep



	def add_single_scatter_cut(self):
		self.Xrun += ' && (Alt$(S2sTot[1],0)<15000.)'
		
		
		
	def add_s1_trig_cut(self):
		Xtrig = ''
		if self.degreeSetting < 0:
			return
		if self.get_timestamp(0) < 1418997600: # 12/19/14 at 9 AM
			Xtrig = '((((TrigLeftEdge[] - S1sPeak[0]) > 0.) && ((TrigLeftEdge[] - S1sPeak[0]) < 50.)) && ((TrigArea[] > (2.2e-8) && TrigArea[] < (2.9e-8)) || (TrigArea[] > (8.2e-8) && TrigArea[] < (10.2e-8)) || (TrigArea[] > (1.50e-7) && TrigArea[] < (1.64e-7)) || (TrigArea[] > 2.2e-7 && TrigArea[] < 2.3e-7)))' #Xtrig3
		elif self.get_timestamp(0) > 1418997600:
			Xtrig = '((((TrigLeftEdge[] - S1sPeak[0]) > 0.) && ((TrigLeftEdge[] - S1sPeak[0]) < 50.)) && ((TrigArea[] > (3.0e-8) && TrigArea[] < (4.8e-8)) || (TrigArea[] > (9.2e-8) && TrigArea[] < (10.6e-8)) || (TrigArea[] > (1.50e-7) && TrigArea[] < (1.7e-7))))' # Xtrig4
		self.Xrun += ' && ' + Xtrig
		
		
		
	def add_cut(self, sCut):
		self.Xrun += ' && ' + sCut
		
		
		
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



	def get_timestamp(self, eventNumber=0, index=0):
		self.lT1[index].GetEntry(eventNumber)
		return self.lT1[index].TimeSec



	def get_livetime(self):
		return self.get_timestamp(self.T1.GetEntries() - 1) - self.get_timestamp(0)



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
		import threading
		if numThreads == 1:
			self.set_event_list(cuts)
		else:
			lock = threading.Lock()
			# set thread tasks
			lThreadTasks = [[] for i in xrange(numThreads)]
			for i in xrange(len(self.lFilenames)):
				lThreadTasks[i%numThreads].append(i)

			# call worker function
			lThreads = [0. for i in xrange(numThreads)]
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