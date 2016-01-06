from math import sqrt

# set constants in this file rather than source code

runNumber = 11
DATA_PATH = './data/run_' + str(runNumber) + '/'
RESULTS_PATH = './results/run_' + str(runNumber) + '/'
CORRECTED_PATH = './corrected_results/run_' + str(runNumber) + '/'




# -------------- define gain correction and rel uncertainty -------------------

def GetGainCorrectionBottomPMT(runNumber, unixtime):
	if runNumber == 10:
	
		if ((unixtime-1.41069584e9) >= 74160 and (unixtime-1.41069584e9) < 1906360):
			return (((unixtime-1.41069584e9)*(-0.0780975672) + 749767.284)/584000.)
		elif ((unixtime-1.41069584e9) >= 1906360 and (unixtime-1.41069584e+09) < 4500160):
			return (((unixtime-1.41069584e9)*(-0.00682007408) + 614268.578)/584000.)
		else:
			return 1.
	
	elif runNumber == 11:

		if ((unixtime-1.41641345e+09) >= 1042584 and (unixtime-1.41641345e+09) < 1405154):
			 return (((unixtime-1.41641345e+09)*(-0.158693682) + 727618.437)/579000.)
		elif ((unixtime-1.41641345e+09) >= 1405154 and (unixtime-1.41641345e+09) < 2597894):
			return (((unixtime-1.41641345e+09)*(-0.0265043046) + 517136.454)/579000.)
		elif ((unixtime-1.41641345e+09) >= 2617454 and (unixtime-1.41641345e+09) < 4082205):
			return (((unixtime-1.41641345e+09)*(-0.117362741) + 1204963.51)/847000.)
		elif ((unixtime-1.41641345e+09) >= 4082205 and (unixtime-1.41641345e+09) < 4860402):
			return (((unixtime-1.41641345e+09)*(-0.0306528978) + 858739.093)/847000.)
		elif ((unixtime-1.41641345e+09) >= 4860402 and (unixtime-1.41641345e+09) < 6652057):
			return (((unixtime-1.41641345e+09)*(-0.00693979741) + 746308.946)/847000.)
		else:
			return 1.

	else:
		print 'Incorrect run number given, please check config.py'
		sys.exit()



def GetGainCorrectionErrorBottomPMT(runNumber, unixtime):
	if runNumber == 10:
	
		if ((unixtime-1.41069584e9) >= 74160 and (unixtime-1.41069584e9) < 1906360):
			return (sqrt(pow((unixtime-1.41069584e+09)*(0.00455971581),2) + pow(6236.27619,2) + 2.*(unixtime-1.41069584e+09)*(-27.0858961))/584000.)
		elif ((unixtime-1.41069584e9) >= 1906360 and (unixtime-1.41069584e+09) < 4500160):
			return (sqrt(pow((unixtime-1.41069584e+09)*(0.00169891684),2) + pow(5824.63918,2) + 2.*(unixtime-1.41069584e+09)*(-9.60028801))/584000.)
		else:
			return 1.
	
	elif runNumber == 11:

		if ((unixtime-1.41641345e+09) >= 1042584 and (unixtime-1.41641345e+09) < 1405154):
			 return (sqrt(pow((unixtime-1.41641345e+09)*(0.097296789),2) + pow(117680.414,2) + 2.*(unixtime-1.41641345e+09)*(-11388.3873))/579000.)
		elif ((unixtime-1.41641345e+09) >= 1405154 and (unixtime-1.41641345e+09) < 2597894):
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0355080231),2) + pow(75117.3648,2) + 2.*(unixtime-1.41641345e+09)*(-2641.21292))/579000.)
		elif ((unixtime-1.41641345e+09) >= 2617454 and (unixtime-1.41641345e+09) < 4082205):
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0113959037),2) + pow(40072.7029,2) + 2.*(unixtime-1.41641345e+09)*(-452.843249))/847000.)
		elif ((unixtime-1.41641345e+09) >= 4082205 and (unixtime-1.41641345e+09) < 4860402):
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0161325496),2) + pow(73013.0196,2) + 2.*(unixtime-1.41641345e+09)*(-1175.4711))/847000.)
		elif ((unixtime-1.41641345e+09) >= 4860402 and (unixtime-1.41641345e+09) < 6652057):
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.00661695924),2) + pow(38105.7583,2) + 2.*(unixtime-1.41641345e+09)*(-250.31314))/847000.)
		else:
			return 1.

	else:
		print 'Incorrect run number given, please check config.py'
		sys.exit()


# gives correction relative to first dataset
def GetHighRateCorrectionBottomPMT(runNumber, unixtime):
	if runNumber == 11:
		if (unixtime >= 1.41745603e+09 and unixtime < 1.4178186e+09):
			return (1.0164418 + (-2.18361852e-07) * (unixtime - 1.41745603e+09))
		elif (unixtime >= 1.4178186e+09 and unixtime < 1.4181663e+09):
			return (0.883167101 + (-1.42727639e-07) * (unixtime - 1.4178186e+09))
		elif (unixtime >= 1.4181663e+09 and unixtime < 1.4186838e+09):
			return (0.842339835 + (-1.62540076e-07) * (unixtime - 1.4181663e+09))
		elif (unixtime >= 1.4186838e+09 and unixtime < 1.4190309e+09):
			return (0.785101604 + (-1.28197152e-07) * (unixtime - 1.4186838e+09))
		elif (unixtime >= 1.4190309e+09 and unixtime < 1.4192814e+09):
			return (1.02296865 + (-1.64819567e-07) * (unixtime - 1.4190309e+09))
		elif (unixtime >= 1.4192814e+09 and unixtime < 1.4196204e+09):
			return (0.979804313 + (-2.15147721e-07) * (unixtime - 1.4192814e+09))
		elif (unixtime >= 1.4196204e+09 and unixtime < 1.4199798e+09):
			return (0.870884983 + (-2.78895215e-07) * (unixtime - 1.4196204e+09))
		elif (unixtime >= 1.4199798e+09 and unixtime < 1.4201946e+09):
			return (0.807355906 + (-1.52944817e-07) * (unixtime - 1.4199798e+09))
		elif (unixtime >= 1.4201946e+09 and unixtime < 1.4212728e+09):
			return (0.722392755 + (-6.14025378e-08) * (unixtime - 1.4201946e+09))
		else:
			return 1.0
	elif runNumber == 10:
		if (unixtime >= 1.4107788e+09 and unixtime < 1.41114493e+09):
			return (1.14694974 + (-3.81524609e-07) * (unixtime - 1.4107788e+09))
		elif (unixtime >= 1.41114493e+09 and unixtime < 1.41159264e+09):
			return (0.983739971 + (-3.93797011e-07) * (unixtime - 1.41114493e+09))
		elif (unixtime >= 1.41159264e+09 and unixtime < 1.41201359e+09):
			return (0.818189627 + (-2.52816072e-07) * (unixtime - 1.41159264e+09))
		elif (unixtime >= 1.41201359e+09 and unixtime < 1.41228348e+09):
			return (0.764060723 + (-2.0719693e-07) * (unixtime - 1.41201359e+09))
		elif (unixtime >= 1.41228348e+09 and unixtime < 1.41236884e+09):
			return (0.638287106 + (-6.35867907e-08) * (unixtime - 1.41228348e+09))
		elif (unixtime >= 1.41236884e+09 and unixtime < 1.41329443e+09):
			return (0.610897867 + (-9.73122426e-08) * (unixtime - 1.41236884e+09))
		elif (unixtime >= 1.41329443e+09 and unixtime < 1.41383939e+09):
			return (0.481255444 + (-1.209793e-07) * (unixtime - 1.41329443e+09))
		elif (unixtime >= 1.41383939e+09 and unixtime < 1.41443875e+09):
			return (0.478083191 + (-6.79189676e-08) * (unixtime - 1.41383939e+09))
		elif (unixtime >= 1.41443875e+09 and unixtime < 1.4150517e+09):
			return (0.378713361 + (-6.11957207e-08) * (unixtime - 1.41443875e+09))
		else:
			return 1.0



# gives correction relative to first dataset
def GetHighRateCorrectionTopPMT(runNumber, unixtime):
	if runNumber == 11:
		if (unixtime >= 1.41745603e+09 and unixtime < 1.4178186e+09):
			return (1.02961367 + (-3.1005902e-07) * (unixtime - 1.41745603e+09))
		elif (unixtime >= 1.4178186e+09 and unixtime < 1.4181663e+09):
			return (0.948384695 + (-3.74109511e-08) * (unixtime - 1.4178186e+09))
		elif (unixtime >= 1.4181663e+09 and unixtime < 1.4186838e+09):
			return (0.866327499 + (-2.34859008e-07) * (unixtime - 1.4181663e+09))
		elif (unixtime >= 1.4186838e+09 and unixtime < 1.4190309e+09):
			return (0.752953525 + (-4.49227681e-08) * (unixtime - 1.4186838e+09))
		elif (unixtime >= 1.4190309e+09 and unixtime < 1.4192814e+09):
			return (0.735994456 + (-1.2944238e-07) * (unixtime - 1.4190309e+09))
		elif (unixtime >= 1.4192814e+09 and unixtime < 1.4196204e+09):
			return (0.686178285 + (-1.07288348e-07) * (unixtime - 1.4192814e+09))
		elif (unixtime >= 1.4196204e+09 and unixtime < 1.4199798e+09):
			return (0.646760604 + (-1.02874732e-07) * (unixtime - 1.4196204e+09))
		elif (unixtime >= 1.4199798e+09 and unixtime < 1.4201946e+09):
			return (0.591000768 + (-5.75954434e-08) * (unixtime - 1.4199798e+09))
		elif (unixtime >= 1.4201946e+09 and unixtime < 1.4212728e+09):
			return (0.572053453 + (-1.42817364e-08) * (unixtime - 1.4201946e+09))
		else:
			return 1.0
	elif runNumber == 10:
		if (unixtime >= 1.4107788e+09 and unixtime < 1.41114493e+09):
			return (1.00401426 + (-1.89912491e-07) * (unixtime - 1.4107788e+09))
		if (unixtime >= 1.41114493e+09 and unixtime < 1.41159264e+09):
			return (0.936529598 + (-3.04203429e-07) * (unixtime - 1.41114493e+09))
		if (unixtime >= 1.41159264e+09 and unixtime < 1.41201359e+09):
			return (0.895075399 + (-6.61046237e-08) * (unixtime - 1.41159264e+09))
		if (unixtime >= 1.41201359e+09 and unixtime < 1.41228348e+09):
			return (0.756071887 + (-1.22737778e-07) * (unixtime - 1.41201359e+09))
		if (unixtime >= 1.41228348e+09 and unixtime < 1.41236884e+09):
			return (0.793128804 + (2.17027804e-07) * (unixtime - 1.41228348e+09))
		if (unixtime >= 1.41236884e+09 and unixtime < 1.41265894e+09):
			return (0.768691432 + (-5.47319914e-08) * (unixtime - 1.41236884e+09))
		if (unixtime >= 1.41265894e+09 and unixtime < 1.41329443e+09):
			return (0.821660756 + (-1.55744654e-08) * (unixtime - 1.41265894e+09))
		if (unixtime >= 1.41329443e+09 and unixtime < 1.41383939e+09):
			return (0.778869818 + (-3.15760023e-08) * (unixtime - 1.41329443e+09))
		if (unixtime >= 1.41383939e+09 and unixtime < 1.41443875e+09):
			return (0.736511035 + (-6.05491311e-08) * (unixtime - 1.41383939e+09))
		if (unixtime >= 1.41443875e+09 and unixtime < 1.4150517e+09):
			return (0.81853132 + (2.38007343e-09) * (unixtime - 1.41443875e+09))
		else:
		   return 1.0
	else:
		print 'Incorrect run number given, please check config.py'
		sys.exit()






# -------------- end definitions of gain correction and rel uncertainty -------------------