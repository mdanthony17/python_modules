import ROOT as root
import sys

class Xenon100_analysis:

	def __init__(self, filesToLoad = []):
		if len(filesToLoad) == 0:
			print 'Need to use a file!'
			exit()
		
		"""
		if fileToLoad[0] == 'n':
			pathToFile = './data/run_11/' + str(fileToLoad)
		else:
			pathToFile = fileToLoad

		try:
			rootFile = root.TFile(pathToFile, 'read')
		except:
			print 'Problem opening file - please check name entered'
			sys.exit(1)
		"""


		self.myTree = root.TChain('T1', 'myTree')
		self.myTree.SetBranchStatus('*', 1)
		self.T2 = root.TChain('T2', 'T2')
		self.T2.SetBranchStatus('*', 1)
		self.T3 = root.TChain('T3', 'T3')
		self.T3.SetBranchStatus('*', 1)

		self.myTree.AddFriend('T2')
		self.myTree.AddFriend('T3')

		for file in filesToLoad:
			if file[0] == 'm':
				pathToFile = '/archive/root/minimum_bias/xenon100/run_12/' + str(file)
			else:
				pathToFile = file

			try:
				print pathToFile
				rootFile = root.TFile(pathToFile, 'read')
			except:
				print 'Problem opening file - please check name entered'
				continue

			self.myTree.AddFile(pathToFile)
			self.T2.AddFile(pathToFile)
			self.T3.AddFile(pathToFile)


		#define helpful aliases
		self.myTree.SetAlias('dt','(S2sPeak[0]-S1sPeak[0])/100.')
		self.myTree.SetAlias('X','S2sPosNn[0][0]')
		self.myTree.SetAlias('Y','S2sPosNn[0][1]')
		self.myTree.SetAlias('R','sqrt(pow(S2sPosNn[0][0],2)+pow(S2sPosNn[0][1],2))')
		self.myTree.SetAlias('czS1sTotBottom','(S1sTotBottom[0]/(0.863098 + (-0.00977873)*Z))')
		self.myTree.SetAlias('ctS1sTotBottom','czS1sTotBottom')
		self.myTree.SetAlias('DriftTime[i]', '(S2sPeak[i]-S1sPeak[0])/100.')
		self.myTree.SetAlias('Z', 'cS2sPosNn[0][2]')
		self.myTree.SetAlias('Z[i]', '-1.8*DriftTime[i]')
		self.myTree.SetAlias('wdt', '59.3888+dt*0.389897-dt*dt*0.000138722-dt*dt*dt*1.01872e-05+dt*dt*dt*dt*5.03672e-08')
		self.myTree.SetAlias('s1asym', '(S1sTotTop[0]-S1sTotBottom[0])/S1sTot[0]')
		self.myTree.SetAlias('s2asym', '(S2sTotTop[0]-S2sTotBottom[0])/S2sTot[0]')
		self.myTree.SetAlias('cs2asym', '(cS2sTotTop[0]-cS2sTotBottom[0])/cS2sTot[0]')
		self.myTree.SetAlias('noisypmts', '((S1s[0][146]>0.35)+(S1s[0][148]>0.35)+(S1s[0][151]>0.35))')
		self.myTree.SetAlias('noisypmts10', '((S1s[0][151]>0.35)+(S1s[0][169]>0.35)+(TMath::Nint(0.4*(S1s[0][166]>0.35)+0.4*(S1s[0][168]>0.35))))')
		self.myTree.SetAlias('ncoin_e', 'Sum$((S1s[0][S1sPmtCoin[0][]]>2)+((S1s[0][S1sPmtCoin[0][]]<=2.&&S1s[0][S1sPmtCoin[0][]]>1)*(S1sEntropy[0][S1sPmtCoin[0][]]<S1s[0][S1sPmtCoin[0][]]+1.5))+((S1s[0][S1sPmtCoin[0][]]<=1.)*(S1sEntropy[0][S1sPmtCoin[0][]]<2.5)))')
		self.myTree.SetAlias('ncoin_e_1', 'Sum$( (S1s[0][S1sPmtCoin[0][]]<0.4806)*(S1sEntropy[0][S1sPmtCoin[0][]]<2.66184) + (0.4806<=S1s[0][S1sPmtCoin[0][]] && S1s[0][S1sPmtCoin[0][]]<=2.08657)*( S1sEntropy[0][S1sPmtCoin[0][]] < (3.52658 - 5.04919*S1s[0][S1sPmtCoin[0][]] + 10.6078*S1s[0][S1sPmtCoin[0][]]**2 - 9.87844*S1s[0][S1sPmtCoin[0][]]**3 + 4.22897*S1s[0][S1sPmtCoin[0][]]**4 - 0.673842*S1s[0][S1sPmtCoin[0][]]**5) ) + (2.08657<S1s[0][S1sPmtCoin[0][]] && S1s[0][S1sPmtCoin[0][]]<15)*(S1sEntropy[0][S1sPmtCoin[0][]]<2.94477) + (S1s[0][S1sPmtCoin[0][]]>=15)*( S1sEntropy[0][S1sPmtCoin[0][]] < ( 7.9527e-01*log10(S1s[0][S1sPmtCoin[0][]]) + 2.0094598 ) ) )') # need to run with 'latest_root' on XeCluster
		self.myTree.SetAlias('ncoin_e_s1_1', 'Alt$(Sum$((S1s[1][S1sPmtCoin[1][]]>2)+((S1s[1][S1sPmtCoin[1][]]<=2.&&S1s[1][S1sPmtCoin[1][]]>1)*(S1sEntropy[1][S1sPmtCoin[1][]]<S1s[1][S1sPmtCoin[1][]]+1.5))+((S1s[1][S1sPmtCoin[1][]]<=1.)*(S1sEntropy[1][S1sPmtCoin[1][]]<2.5))),0)')
		self.myTree.SetAlias('ncoin_e_s1_2', 'Alt$(Sum$((S1s[2][S1sPmtCoin[2][]]>2)+((S1s[2][S1sPmtCoin[2][]]<=2.&&S1s[2][S1sPmtCoin[2][]]>1)*(S1sEntropy[2][S1sPmtCoin[2][]]<S1s[2][S1sPmtCoin[2][]]+1.5))+((S1s[2][S1sPmtCoin[2][]]<=1.)*(S1sEntropy[2][S1sPmtCoin[2][]]<2.5))),0)')
		self.myTree.SetAlias('ncoin_e_s1_3', 'Alt$(Sum$((S1s[3][S1sPmtCoin[3][]]>2)+((S1s[3][S1sPmtCoin[3][]]<=2.&&S1s[3][S1sPmtCoin[3][]]>1)*(S1sEntropy[3][S1sPmtCoin[3][]]<S1s[3][S1sPmtCoin[3][]]+1.5))+((S1s[3][S1sPmtCoin[3][]]<=1.)*(S1sEntropy[3][S1sPmtCoin[3][]]<2.5))),0)')
		self.myTree.SetAlias('ncoin_e_s1_4', 'Alt$(Sum$((S1s[4][S1sPmtCoin[4][]]>2)+((S1s[4][S1sPmtCoin[4][]]<=2.&&S1s[4][S1sPmtCoin[4][]]>1)*(S1sEntropy[4][S1sPmtCoin[4][]]<S1s[4][S1sPmtCoin[4][]]+1.5))+((S1s[4][S1sPmtCoin[4][]]<=1.)*(S1sEntropy[4][S1sPmtCoin[4][]]<2.5))),0)')
		self.myTree.SetAlias('PL013', '(-2*(S1PatternLikelihood[0]+S1PatternLikelihood[1]+S1PatternLikelihood[3]))')
		self.myTree.SetAlias('mean_width', '(TMath::Sqrt((1.18791e+02-4.53908e+00*log(S2sTot[0])+1.93864e-01*sqrt(S2sTot[0])-2.64963e-04*S2sTot[0])^2+(-2.93710e+02+9.43304e+01*log(S2sTot[0])-6.49995e+00*sqrt(S2sTot[0])+3.02096e-02*S2sTot[0])*dt))')
		self.myTree.SetAlias('sigma_width', '(4.65009e+01-8.24143e+00*log(S2sTot[0])+5.81395e-01*sqrt(S2sTot[0])-2.96082e-03*S2sTot[0]+(6.92605e-02+2.37447e-02*log(S2sTot[0])-5.77711e-03*sqrt(S2sTot[0])+3.71555e-05*S2sTot[0])*dt)')
		self.myTree.SetAlias('sigma_flattened_width', '6.77404e+01-1.01215e+01*log(S2sTot[0])+2.97558e-01*sqrt(S2sTot[0])-1.08131e+01*(1./exp(1.29359e-03*S2sTot[0])) + (3.82098e-02 + 9.63804e-02*exp( -5.15204e-04*S2sTot[0]))*dt')
		self.myTree.SetAlias('sigma_flattened_width_11', '(6.77404e+01-1.01215e+01*log(S2sTot[0])+2.97558e-01*sqrt(S2sTot[0])-1.08131e+01*(1./exp(1.29359e-03*S2sTot[0])) + (3.82098e-02 + 9.63804e-02*exp( -5.15204e-04*S2sTot[0]))*dt)*(S2sTot[0]<5000.)+(6.77404e+01-1.01215e+01*log(5000.)+2.97558e-01*sqrt(5000.)-1.08131e+01*(1./exp(1.29359e-03*5000.)) + (3.82098e-02 + 9.63804e-02*exp( -5.15204e-04*5000.))*dt)*(S2sTot[0]>=5000.)')
		
		
		# standard Xenon100 cuts
		Xsignalnoise5_corrected = '((log10(S2sTot[0])<2.77 && log10((S1sTot[0]+S2sTot[0])/TMath::Max(AreaTot-S1sTot[0]-S2sTot[0],0.00001))>0.0 && AreaTot>0.0) || (log10(S2sTot[0])>=2.77 && log10((S1sTot[0]+S2sTot[0])/TMath::Max(AreaTot-S1sTot[0]-S2sTot[0],0.00001))>TMath::Min(0.23436*log10(S2sTot[0])^3-2.77487*log10(S2sTot[0])^2+11.2553*log10(S2sTot[0])-14.8667,0.8) && AreaTot>0.0))'
		Xs1width0 = '(S1sLowWidth[0]>2.6)'
		Xentropy1 = 'ncoin_e_1>1' # probably need alias for this
		Xs2peaks2 = '(S2sTot[0] > 150)'
		Xs1coin2 = '(S1sCoin[0]>(1+noisypmts10))' # probably need alias for this
		
		Xhighlog1 = '(log10(cS2sTotBottom[0]/cS1sTot[0])<3.1)'
		Xs2single3 = '(Alt$(S2sTot[1],0)<(70+(S2sTot[0]-300)/100.)) '
		Xs1single5 = '((ncoin_e_s1_1 < 2 || (S2sPeak[0]-Alt$(S1sPeak[1],-18500)) > 18500 || log10(cS2sTot[0]/Alt$(cS1sTot[1],1.e-4)) > 3.2) && (ncoin_e_s1_2 < 2 || (S2sPeak[0]-Alt$(S1sPeak[2],-18500)) > 18500 || log10(cS2sTot[0]/Alt$(cS1sTot[2],1.e-4)) > 3.2) && (ncoin_e_s1_3 < 2 || (S2sPeak[0]-Alt$(S1sPeak[3],-18500)) > 18500 || log10(cS2sTot[0]/Alt$(cS1sTot[3],1.e-4)) > 3.2) && (ncoin_e_s1_4 < 2))' # probably need alias for this
		Xveto2 = '(S1sTotOutside[0]<0.35)'
		Xs2width11 = '(cfS2sLowWidth[0]>-2.*sigma_flattened_width_11 && cfS2sLowWidth[0]<2.*sigma_flattened_width_11)' # probably need alias for this
		
		XPL013_97 = '(PL013 < -16.9181 + 27.3756 * pow(S1sTot[0],0.5) + -1.73185 * S1sTot[0] + 0.041431 * pow(S1sTot[0],1.5))' # probably need alias for this
		Xposrec1 = '(sqrt((pow((S2sPosNn[0][0]-S2sPosSvm[0][0]),2)+pow((S2sPosNn[0][1]-S2sPosSvm[0][1]),2)) + (pow((S2sPosNn[0][0]-S2sPosChi2[0][0]),2)+pow((S2sPosNn[0][1]-S2sPosChi2[0][1]),2)))<7)'
		Xs2chisquare1 = '((S2sPosNn[0][3]/(S2sCoinTop[0]-1))<6)'
		Xlownoise0_m = 'cs2asym<0.25 && cs2asym>-0.2' # probably need alias for this
		Xs2peakpos0 = '(S2sPeak[0]/100) > 178'
		
		Xtime_diff_50 = '(TimeSec-PrevTimeSec)*1e6+(TimeMicroSec-PrevTimeMicroSec)>50000'
		X34kg2 = '(TMath::Power(TMath::Abs(-Z-152.)/126.8,2.7)+TMath::Power((X*X+Y*Y)/17500.,2.7)<1)' # probably need alias for this
		
		# list of still needed aliases:
		# ncoin_e_1, noisypmts10, ncoin_e_s1_4, sigma_flattened_width_11,
		# PL013, cs2asym, Z
		# they are on same page as cuts
		

		self.Xrun = '(S1sCoin[0] != -1)' #add a cut so that add statements work
		
		self.Xrun = self.Xrun + ' && ' + Xsignalnoise5_corrected
		self.Xrun = self.Xrun + ' && ' + Xs1width0
		self.Xrun = self.Xrun + ' && ' + Xentropy1
		self.Xrun = self.Xrun + ' && ' + Xs2peaks2
		self.Xrun = self.Xrun + ' && ' + Xs1coin2
		
		self.Xrun = self.Xrun + ' && ' + Xhighlog1
		self.Xrun = self.Xrun + ' && ' + Xs2single3
		self.Xrun = self.Xrun + ' && ' + Xs1single5
		#self.Xrun = self.Xrun + ' && ' + Xveto2
		self.Xrun = self.Xrun + ' && ' + Xs2width11
		
		self.Xrun = self.Xrun + ' && ' + XPL013_97
		self.Xrun = self.Xrun + ' && ' + Xposrec1
		self.Xrun = self.Xrun + ' && ' + Xs2chisquare1
		self.Xrun = self.Xrun + ' && ' + Xlownoise0_m
		self.Xrun = self.Xrun + ' && ' + Xs2peakpos0
		
		self.Xrun = self.Xrun + ' && ' + Xtime_diff_50
		self.Xrun = self.Xrun + ' && ' + X34kg2
		
		



	def get_tree(self):
		return self.myTree



	def add_dt_cut(self, lowdt = 2., highdt = 13.):
		Xdt = '((dt > ' + str(lowdt) + ') && (dt < '+ str(highdt) + '))'
		self.Xrun = self.Xrun + '&&' + Xdt



	def add_radius_cut(self, lowRadius = 0., highRadius = 20.):
		Xradius = '((R > ' + str(lowRadius) + ') && (R < '+ str(highRadius) + '))'
		self.Xrun = self.Xrun + '&&' + Xradius
		


	def add_eDep_cut(self, lowEnergy = -2., highEnergy = 35.):
		Xedep = '((eDep > '+str(lowEnergy)+') && (eDep < '+str(highEnergy)+'))'
		self.Xrun = self.Xrun + '&&' + Xedep
		
		
		
	def reset_cuts(self):
		self.Xrun = 'S1sCoin[0] > 0'
		



	def get_cuts(self):
		return self.Xrun



	def get_timestamp(self, eventNumber):
		self.myTree.GetEntry(eventNumber)
		return self.myTree.TimeSec



	def get_livetime(self):
		return self.get_timestamp(self.myTree.GetEntries() - 1) - self.get_timestamp(0)



	def set_event_list(self, eventListName = None, cuts = None):
		if cuts == None:
			cuts = self.Xrun
		if eventListName == None:
			eventListName = 'eList'
		self.reset_event_list()
		self.eList = root.TEventList(eventListName)
		self.myTree.Draw('>>' + eventListName, root.TCut(cuts), '') #second arg should be cuts
		self.myTree.SetEventList(self.eList)
		
	
	
	def reset_event_list(self):
		self.myTree.SetEventList(0)
		if 'eList' in locals():
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



	def save_event_list(self, fileName = ''):
		if fileName == '':
			fileName = 'eList.root'
		fEventList = root.TFile(fileName, 'recreate')
		self.get_event_list().Write(fileName[:-5])
		fEventList.Close()



	def load_event_list(self, fileName = ''):
		if fileName == '':
			fileName = 'eList.root'
		eventListName = fileName[:-5]
		fEventList = root.TFile(fileName, 'read')
		self.eList = root.TEventList(fEventList.Get(eventListName))
		self.myTree.SetEventList(self.eList)


	def make_reduced_tree(self, fileName = ''):
		if fileName == '':
			fileName = 'reduced_tree.root'

		outputFile = root.TFile('./reduced_trees/' + fileName, 'recreate')
		newT1 = root.TTree()
		newT2 = root.TTree()
		newT3 = root.TTree()
		

		newT1 = self.myTree.CloneTree(0)
		newT2 = self.T2.CloneTree(0)
		newT3 = self.T3.CloneTree(0)

		entryNumber = 0

		for i in xrange(self.eList.GetN()):
			entryNumber = self.myTree.GetEntryNumber(i)

			self.myTree.GetEvent(entryNumber)
			newT1.Fill()

			self.T2.GetEvent(entryNumber)
			newT2.Fill()

			self.T3.GetEvent(entryNumber)
			newT3.Fill()

		newT1.Write()
		newT2.Write()
		newT3.Write()

		outputFile.Close()




if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'Need file that program will convert to list'
		print 'and name of event list to use'
		print 'python Xenon100_analysis.py <filename> <event list name (no .root extension)>'
	
	print 'Need to run after script "latest_root"'

	fileName = sys.argv[1]
	savedFileName = sys.argv[2]
	lFiles = []

	fFilesToUse = open(fileName, 'r')

	for line in fFilesToUse:
		line = line[:-1]
		if len(line) > 2:
			lFiles.append('mb_' + line + '.root')

	print 'Finished creating list'

	run = Xenon100_analysis(lFiles)

	print 'Total number of entries: ' + str(run.get_tree().GetEntries())

	print 'About to create event list'
	print 'This may take several minutes...'
	run.set_event_list()

	print 'Event list finished'
	print 'Entries after cuts: ' + str(run.get_event_list().GetN())

	run.save_event_list('el_' + savedFileName + '.root')
	run.make_reduced_tree('rt_' + savedFileName + '.root')

	print 'Event list created in current directory: ' + str('el_' + savedFileName + + '.root')
	print 'Reduced tree created in current directory: ' + str('rt_' + savedFileName + + '.root')
