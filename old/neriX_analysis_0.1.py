import ROOT as root
import sys

class neriX_analysis:
	__GERMANIUM_SLOPE = '692.905899'
	__GERMANIUM_INTERCEPT = '1.67455'

	def __init__(self, fileToLoad = '', run = 11):
		if fileToLoad == '':
			print 'Need to use a file!'
			exit()
		
		
		self.runNumber = run

		if fileToLoad[0] == 'n':
			pathToFile = './data/run_' + str(self.runNumber) + '/' + str(fileToLoad)
		else:
			pathToFile = fileToLoad

		try:
			rootFile = root.TFile(pathToFile, 'read')
		except:
			print 'Problem opening file - please check name entered'
			sys.exit(1)


		self.myTree = root.TChain('T1', 'myTree')
		self.myTree.SetBranchStatus('*', 1)
		self.T2 = root.TChain('T2', 'T2')
		self.T2.SetBranchStatus('*', 1)
		self.T3 = root.TChain('T3', 'T3')
		self.T3.SetBranchStatus('*', 1)

		self.myTree.AddFriend('T2')
		self.myTree.AddFriend('T3')

		self.myTree.AddFile(pathToFile)
		self.T2.AddFile(pathToFile)
		self.T3.AddFile(pathToFile)


		#define helpful aliases
		self.myTree.SetAlias('dt','(S2sPeak[0]-S1sPeak[0])/100.')
		self.myTree.SetAlias('X','S2sPosFann[0][0]')
		self.myTree.SetAlias('Y','S2sPosFann[0][1]')
		self.myTree.SetAlias('R','sqrt(pow(S2sPosFann[0][0],2)+pow(S2sPosFann[0][1],2))')
		self.myTree.SetAlias('czS1sTotBottom','(S1sTotBottom[0]/(0.863098 + (-0.00977873)*Z))')
		self.myTree.SetAlias('ctS1sTotBottom','czS1sTotBottom')
		self.myTree.SetAlias('s1asym', '(S1sTotTop[0]-S1sTotBottom[0])/S1sTot[0]')
		self.myTree.SetAlias('s2asym', '(S2sTotTop[0]-S2sTotBottom[0])/S2sTot[0]')

		#will need to change these constants depending on Ge Calibratio
		self.myTree.SetAlias('GeEnergy',self.__GERMANIUM_SLOPE + '*GeHeight[0] + ' + self.__GERMANIUM_INTERCEPT)
		self.myTree.SetAlias('eDep','661.657 - GeEnergy')
		
		self.Xrun = '(S1sCoin[0] != -1)' #add a cut so that add statements work



	def get_tree(self):
		return self.myTree



	def add_dt_cut(self, lowdt = 2., highdt = 13.):
		Xdt = '((dt > ' + str(lowdt) + ') && (dt < '+ str(highdt) + '))'
		self.Xrun = self.Xrun + ' && ' + Xdt



	def add_radius_cut(self, lowRadius = 0., highRadius = 20.):
		Xradius = '((R > ' + str(lowRadius) + ') && (R < '+ str(highRadius) + '))'
		self.Xrun = self.Xrun + ' && ' + Xradius
		


	def add_eDep_cut(self, lowEnergy = -2., highEnergy = 35.):
		Xedep = '((eDep > '+str(lowEnergy)+') && (eDep < '+str(highEnergy)+'))'
		self.Xrun = self.Xrun + ' && ' + Xedep
		
		
		
	def add_s1_trig_cut(self):
		Xtrig = ''
		if self.get_timestamp(0) < 1418997600: # 12/19/14 at 9 AM
			Xtrig = '((((TrigLeftEdge[] - S1sPeak[0]) > 0.) && ((TrigLeftEdge[] - S1sPeak[0]) < 50.)) && ((TrigArea[] > (2.2e-8) && TrigArea[] < (2.9e-8)) || (TrigArea[] > (8.2e-8) && TrigArea[] < (10.2e-8)) || (TrigArea[] > (1.50e-7) && TrigArea[] < (1.64e-7)) || (TrigArea[] > 2.2e-7 && TrigArea[] < 2.3e-7)))' #Xtrig3
		elif self.get_timestamp(0) > 1418997600:
			Xtrig = '((((TrigLeftEdge[] - S1sPeak[0]) > 0.) && ((TrigLeftEdge[] - S1sPeak[0]) < 50.)) && ((TrigArea[] > (3.0e-8) && TrigArea[] < (4.8e-8)) || (TrigArea[] > (9.2e-8) && TrigArea[] < (10.6e-8)) || (TrigArea[] > (1.50e-7) && TrigArea[] < (1.7e-7))))' # Xtrig4
		self.Xrun += ' && ' + Xtrig
		
		
		
	def add_cut(self, sCut):
		self.Xrun += ' && ' + sCut
		
		
		
	def reset_cuts(self):
		self.Xrun = 'S1sCoin[0] > 0'
		



	def get_cuts(self):
		return self.Xrun



	def get_timestamp(self, eventNumber):
		self.myTree.GetEntry(eventNumber)
		return self.myTree.TimeSec



	def get_livetime(self):
		return self.get_timestamp(self.myTree.GetEntries() - 1) - self.get_timestamp(0)



	def set_event_list(self, cuts = None):
		if cuts == None:
			cuts = self.Xrun
		self.reset_event_list()
		self.eList = root.TEventList('eList')
		self.myTree.Draw('>>eList', root.TCut(cuts), '') #second arg should be cuts
		self.myTree.SetEventList(self.eList)
		
	
	
	def reset_event_list(self):
		self.myTree.SetEventList(0)
		if 'elist' in locals():
			self.eList.Clear()
			self.eList.Reset()
			self.eList.SetDirectory(0)
			#del self.eList



	def get_event_list(self):
		return self.eList



	def reset_tree(self):
		if 'myTree' in locals():
			del self.myTree
		if 'T2' in locals():
			del self.T2
		if 'T3' in locals():
			del self.T3






#test = neriX_analysis('nerix_140914_1631.root')
#test.get_timestamp(1)
#print test.get_livetime()