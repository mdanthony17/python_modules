run_files = {}

sPathToData = '/Users/Matt/Desktop/Xenon/neriX/data/'
sPathToProcessedData = '/Users/Matt/Desktop/Xenon/neriX/calibrations/pmt_gain/processed_data/'
sPathToGraphs = '/Users/Matt/Desktop/Xenon/neriX/calibrations/pmt_gain/gain_graphs/'
sPathToHists = '/Users/Matt/Desktop/Xenon/neriX/calibrations/pmt_gain/gain_hists/'

lHistParameters = (100, -1e6, 4.0e6)
lGraphRangeY = [0.7e6, 2.0e6]

dChannelsAtLightLevels = {0:[],
						  1:[17],
					      2:[4, 10],
						  3:[2, 3, 7, 8, 9, 12, 13, 14],
						  4:[1, 5, 6, 11, 15, 16]}

lLightLevels = [0, 1, 2, 3, 4]

dLightLevels = {1:4,
				2:3,
				3:3,
				4:2,
				5:4,
				6:4,
				7:3,
				8:3,
				9:3,
				10:2,
				11:4,
				12:3,
				13:3,
				14:3,
				15:4,
				16:4,
				17:1}

# set runs to use here
runsInUse = [14, 15]

for run in runsInUse:
	run_files[run] = []

# run_files[run][index][light level] = name of file
for run in run_files:
	if run == 14:
		run_files[run].append({0:'nerix_150819_1511.root',
							   1:'nerix_150819_1515.root',
							   2:'nerix_150819_1524.root',
							   3:'nerix_150819_1528.root',
							   4:'nerix_150819_1532.root'})

	elif run == 15:
		run_files[run].append({0:'nerix_150831_1302.root',
							   1:'nerix_150831_1310.root',
							   2:'nerix_150831_1317.root',
							   3:'nerix_150831_1359.root',
							   4:'nerix_150831_1406.root'})

		run_files[run].append({0:'nerix_150903_0948.root',
							   1:'nerix_150903_0956.root',
							   2:'nerix_150903_1003.root',
							   3:'nerix_150903_1016.root',
							   4:'nerix_150903_1023.root'})

		#run_files[run].append({0:'nerix_150903_1717.root',
		#					   1:'nerix_150903_1724.root'})

		run_files[run].append({0:'nerix_150904_1048.root',
							   1:'nerix_150904_1103.root'})

		run_files[run].append({0:'nerix_150909_0918.root',
							   1:'nerix_150909_0924.root',
							   2:'nerix_150909_0932.root',
							   3:'nerix_150909_0940.root',
							   4:'nerix_150909_0947.root'})

		run_files[run].append({0:'nerix_150916_1502.root',
							   1:'nerix_150916_1511.root',
							   2:'nerix_150916_1520.root',
							   3:'nerix_150916_1528.root',
							   4:'nerix_150916_1535.root'})

		run_files[run].append({0:'nerix_150921_1053.root',
							   1:'nerix_150921_1114.root',
							   2:'nerix_150921_1120.root',
							   3:'nerix_150921_1125.root',
							   4:'nerix_150921_1130.root'})

		run_files[run].append({0:'nerix_150928_1232.root',
							   1:'nerix_150928_1054.root',
							   2:'nerix_150928_1102.root',
							   3:'nerix_150928_1109.root',
							   4:'nerix_150928_1123.root'})

		run_files[run].append({0:'nerix_151001_1413.root',
							   1:'nerix_151001_1420.root'})

		run_files[run].append({0:'nerix_151004_1059.root',
							   1:'nerix_151004_1110.root',
							   2:'nerix_151004_1119.root',
							   3:'nerix_151004_1133.root',
							   4:'nerix_151004_1142.root'})

		run_files[run].append({0:'nerix_151006_0820.root',
							   1:'nerix_151006_0832.root'})

		run_files[run].append({0:'nerix_151013_1003.root',
							   1:'nerix_151013_1013.root',
							   2:'nerix_151013_1022.root',
							   3:'nerix_151013_1031.root',
							   4:'nerix_151013_1040.root'})

		run_files[run].append({0:'nerix_151026_1018.root',
							   1:'nerix_151026_1023.root',
							   2:'nerix_151026_1031.root',
							   3:'nerix_151026_1040.root',
							   4:'nerix_151026_1048.root'})





