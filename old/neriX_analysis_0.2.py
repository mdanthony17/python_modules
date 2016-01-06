import ROOT as root
import sys
from rootpy import stl
from rootpy.io import File
from rootpy.tree import Tree, TreeModel, TreeChain
import neriX_config, neriX_datasets

# MUST INCLUDE TYPES TO AVOID SEG FAULT
stl.vector(stl.vector('float'))
stl.vector(stl.vector('int'))

class neriX_analysis:
	__GERMANIUM_SLOPE = '693.505789'
	__GERMANIUM_INTERCEPT = '1.717426'

	def __init__(self, fileToLoad = ''):
		if fileToLoad == '':
			print 'Need to use a file!'
			sys.exit()
		
		# check neriX_datasets for file and parameters
		
		if fileToLoad[-5:] != '.root':
			fileToLoad += '.root'
			
		self.filename = fileToLoad[-22:] # just grab neriX portion
		
		PARAMETERS_INDEX = neriX_datasets.PARAMETERS_INDEX
		ANODE_INDEX = neriX_datasets.ANODE_INDEX
		CATHODE_INDEX = neriX_datasets.CATHODE_INDEX
		DEGREE_INDEX = neriX_datasets.DEGREE_INDEX

		dRunFiles = neriX_datasets.run_files
		lRuns = neriX_datasets.runsInUse
		
		lParameters = None
		for run in lRuns:
			try:
				lParameters = dRunFiles[run][self.filename]
				currentRun = run
				break
			except KeyError:
				continue
			
		if not lParameters:
			print 'File does not appear in neriX_datasets.py'
			print 'Please check if the file exists and that your spelling is correct'
			sys.exit()

		
		self.runNumber = currentRun

		if fileToLoad[0] == 'n' or fileToLoad[0] == 'c':
			pathToFile = neriX_config.pathToData + 'run_' + str(self.runNumber) + '/' + str(fileToLoad)
		else:
			pathToFile = fileToLoad
		
		self.rootFile = File(pathToFile, 'read')
		
		if self.rootFile.keys() == []:
			print 'Problem opening file - please check name entered'
			print 'Entered: ' + pathToFile
			sys.exit(1)

		self.anodeSetting = dRunFiles[self.runNumber][self.filename][ANODE_INDEX]
		self.cathodeSetting = dRunFiles[self.runNumber][self.filename][CATHODE_INDEX]
		self.degreeSetting = dRunFiles[self.runNumber][self.filename][DEGREE_INDEX]

		# load trees

		self.T1 = self.rootFile.T1
		self.T2 = self.rootFile.T2
		self.T3 = self.rootFile.T3
		try:
			self.T4 = self.rootFile.T4
		except AttributeError:
			self.T4 = None
		
		self.T1.SetName('T1_' + self.filename)
		self.T2.SetName('T2_' + self.filename)
		self.T3.SetName('T3_' + self.filename)
		if self.T4:
			self.T4.SetName('T4_' + self.filename)

		self.T1.create_buffer()
		self.T2.create_buffer()
		self.T3.create_buffer()
		if self.T4:
			self.T4.create_buffer()

		self.T1.AddFriend('T2_' + self.filename)
		self.T1.AddFriend('T3_' + self.filename)
		if self.T4:
			self.T1.AddFriend('T4_' + self.filename)



		#define helpful aliases
		
		self.T1.SetAlias('dt','(S2sPeak[0]-S1sPeak[0])/100.')
		self.T1.SetAlias('X','S2sPosFann[0][0]')
		self.T1.SetAlias('Y','S2sPosFann[0][1]')
		if self.T4:
			self.T1.SetAlias('R','sqrt(pow(ctNNPos[0],2)+pow(ctNNPos[1],2))')
		else:
			self.T1.SetAlias('R','sqrt(pow(S2sPosFann[0][0],2)+pow(S2sPosFann[0][1],2))')
		self.T1.SetAlias('czS1sTotBottom','(S1sTotBottom[0]/(0.863098 + (-0.00977873)*Z))')
		self.T1.SetAlias('ctS1sTotBottom','czS1sTotBottom')
		self.T1.SetAlias('s1asym', '(S1sTotTop[0]-S1sTotBottom[0])/S1sTot[0]')
		self.T1.SetAlias('s2asym', '(S2sTotTop[0]-S2sTotBottom[0])/S2sTot[0]')
		
		if self.cathodeSetting == 0.345:
			self.T1.SetAlias('Z','-1.511*(S2sPeak[0]-S1sPeak[0])/100')
		elif self.cathodeSetting == 1.054:
			self.T1.SetAlias('Z','-1.717*(S2sPeak[0]-S1sPeak[0])/100')
		elif self.cathodeSetting == 2.356:
			self.T1.SetAlias('Z','-1.958*(S2sPeak[0]-S1sPeak[0])/100')
		elif self.cathodeSetting == 5.500:
			self.T1.SetAlias('Z','-2.208*(S2sPeak[0]-S1sPeak[0])/100')
		else:
			print 'Incorrect field entered - cannot correct Z'

		#will need to change these constants depending on Ge Calibratio
		self.T1.SetAlias('GeEnergy',self.__GERMANIUM_SLOPE + '*GeHeight[0] + ' + self.__GERMANIUM_INTERCEPT)
		self.T1.SetAlias('eDep','661.657 - GeEnergy')
		
		self.Xrun = '(EventId != -1)' #add a cut so that add statements work
		
		# create event list
		self.eList = root.TEventList('eList_' + self.filename)
		root.SetOwnership(self.eList, True)
		




	def get_tree(self):
		return self.T1
	
	
	
	def get_run(self):
		return self.runNumber
	
	
	
	def get_filename(self):
		return self.filename
	
	
	
	def get_T1(self):
		return self.T1
	
	
	
	def get_T2(self):
		return self.T2
			
	
	
	def get_T3(self):
		return self.T3
	
	
	
	def get_T4(self):
		return self.T4



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
		self.Xrun = 'S1sCoin[0] > 0'



	def get_degree_setting(self):
		return self.degreeSetting



	def get_cathode_setting(self):
		return self.cathodeSetting



	def get_anode_setting(self):
		return self.anodeSetting



	def get_cuts(self):
		return self.Xrun



	def get_timestamp(self, eventNumber=0):
		self.T1.GetEntry(eventNumber)
		return self.T1.TimeSec



	def get_livetime(self):
		return self.get_timestamp(self.T1.GetEntries() - 1) - self.get_timestamp(0)



	def set_event_list(self, cuts = None):
		if cuts == None:
			cuts = self.Xrun
		#self.reset_event_list()
		print 'Original elements in tree: ' + str(self.T1.GetEntriesFast())
		root.TTree.Draw(self.T1, '>>eList_' + self.filename, root.TCut(cuts), '') #second arg should be cuts
		self.T1.SetEventList(self.eList)
		print 'Elements after cuts: ' + str(self.eList.GetN())
		
	
	
	def reset_event_list(self):
		self.T1.SetEventList(0)
		if ('elist_' + self.filename) in locals():
			self.eList.Clear()
			self.eList.Reset()
			self.eList.SetDirectory(0)
			#del self.eList



	def get_event_list(self):
		return self.eList



	def reset_tree(self):
		if 'T1' in locals():
			del self.T1
		if 'T2' in locals():
			del self.T2
		if 'T3' in locals():
			del self.T3






#test = neriX_analysis('nerix_140914_1631.root')
#test.get_timestamp(1)
#print test.get_livetime()