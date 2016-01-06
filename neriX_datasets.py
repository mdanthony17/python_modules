run_files = {}

PARAMETERS_INDEX = 0
# once inside voltage Index
ANODE_INDEX = 0
CATHODE_INDEX = 1
DEGREE_INDEX = 2 #-6 for bkg, -5 for na, -4 minitron, -3 for TDC, -2 (for co cal), -1 (for cs cal), 0 deg, or 25 deg


# set runs to use here
runsInUse = [10, 11, 13, 14, 15]

for run in runsInUse:
	run_files[run] = {}

for run in run_files:
	if run == 10:
		# run_10
		
		#Cs calibration files
		run_files[run]['nerix_140915_1043.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_140917_1010.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_140924_1007.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_140924_1158.root'] = (4.50, 5.500, -1)
		run_files[run]['nerix_140929_1115.root'] = (4.50, 5.500, -1)
		
		run_files[run]['nerix_140929_1212.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_140930_0925.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_140930_1029.root'] = (4.50, 5.500, -1)
		run_files[run]['nerix_140930_1127.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_140930_1626.root'] = (4.50, 2.356, -1)
		
		run_files[run]['nerix_141001_1102.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_141003_1120.root'] = (4.50, 1.054, -1) #missing
		run_files[run]['nerix_141003_1232.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141006_0951.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141008_1025.root'] = (4.50, 1.054, -1)
		
		run_files[run]['nerix_141010_1211.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141010_1318.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141013_1034.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141014_1024.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141014_1414.root'] = (4.50, 1.054, -1)
		
		run_files[run]['nerix_141015_1155.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141021_1135.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141022_1127.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_141022_1150.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_141024_1556.root'] = (4.50, 0.345, -1)
		
		run_files[run]['nerix_141027_1041.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141103_1101.root'] = (4.50, 5.500, -1)
		run_files[run]['nerix_141103_1119.root'] = (4.50, 5.500, -1)
		
		
		#Co calibrations
		run_files[run]['nerix_140917_1027.root'] = (4.50, 2.356, -2)
		run_files[run]['nerix_140924_1052.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_140924_1252.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_140929_1144.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_140929_1330.root'] = (4.50, 0.345, -2)
	
		run_files[run]['nerix_141001_1012.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_141003_1301.root'] = (4.50, 2.356, -2)
		run_files[run]['nerix_141003_1143.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_141006_1029.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_141008_1058.root'] = (4.50, 1.054, -2)
		
		run_files[run]['nerix_141013_1112.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_141015_1052.root'] = (4.50, 2.356, -2)
		run_files[run]['nerix_141022_1241.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_141024_1731.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_141027_1139.root'] = (4.50, 2.356, -2)
		
		run_files[run]['nerix_141029_1213.root'] = (4.50, 5,500, -2)
		run_files[run]['nerix_141103_1203.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_141104_1433.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_141104_1531.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_141104_1633.root'] = (4.50, 2.356, -2)
		
		run_files[run]['nerix_141105_1259.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_141106_1258.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_141106_1543.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_141107_1138.root'] = (4.50, 5.500, -2)
	
	
		# coincidence files
		run_files[run]['nerix_140915_1457.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140915_1742.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140916_0824.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140916_1447.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140916_1623.root'] = (4.50, 2.356, 0)

		run_files[run]['nerix_140917_1234.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140917_1644.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140918_0730.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140918_1245.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140918_1801.root'] = (4.50, 2.356, 0)

		run_files[run]['nerix_140919_0921.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_140919_1341.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_140919_1719.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_140920_1101.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_140921_1053.root'] = (4.50, 1.054, 0)

		run_files[run]['nerix_140922_0710.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_140922_1006.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_140922_2034.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_140923_1001.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_140924_1803.root'] = (4.50, 5.500, 0)

		run_files[run]['nerix_140925_0739.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_140925_1809.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_140926_0717.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_140926_1409.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_140927_0035.root'] = (4.50, 5.500, 0)

		run_files[run]['nerix_140928_0727.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_140929_1458.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_140929_1720.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_140930_0717.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_140930_1249.root'] = (4.50, 0.345, 0)

		run_files[run]['nerix_140930_1736.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_141001_1227.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_141001_1646.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_141002_1007.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_141002_1133.root'] = (4.50, 0.345, 0)

		run_files[run]['nerix_141002_1658.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_141003_0938.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_141003_1740.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141004_0948.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141005_1005.root'] = (4.50, 1.054, 0)

		run_files[run]['nerix_141005_2254.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141006_1726.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141006_1758.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141007_0831.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141007_1137.root'] = (4.50, 1.054, 25)

		run_files[run]['nerix_141007_1506.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141007_1506.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141008_1640.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141009_1810.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141010_0831.root'] = (4.50, 1.054, 25)

		run_files[run]['nerix_141010_1054.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141010_1356.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141010_1440.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141011_1128.root'] = (4.50, 1.054, 25)
		run_files[run]['nerix_141012_0935.root'] = (4.50, 1.054, 25)

		run_files[run]['nerix_141014_1046.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141014_1231.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141014_1512.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141015_0719.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141017_1019.root'] = (4.50, 2.356, 25)

		run_files[run]['nerix_141017_1406.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141017_1458.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141017_1713.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141018_1252.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141019_0009.root'] = (4.50, 2.356, 25)

		run_files[run]['nerix_141019_1449.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141020_1810.root'] = (4.50, 0.345, 25)
		run_files[run]['nerix_141021_1027.root'] = (4.50, 0.345, 25)
		run_files[run]['nerix_141021_1356.root'] = (4.50, 0.345, 25)
		run_files[run]['nerix_141021_1611.root'] = (4.50, 0.345, 25)

		run_files[run]['nerix_141021_1722.root'] = (4.50, 0.345, 25)
		run_files[run]['nerix_141023_1240.root'] = (4.50, 0.345, 25)
		run_files[run]['nerix_141024_0826.root'] = (4.50, 0.345, 25)
		run_files[run]['nerix_141024_1802.root'] = (4.50, 0.345, 25)
		run_files[run]['nerix_141025_1327.root'] = (4.50, 2.356, 25)

		run_files[run]['nerix_141026_1138.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141027_1230.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141027_1639.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141028_1050.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141029_1425.root'] = (4.50, 5.500, 25)

		run_files[run]['nerix_141030_0846.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141031_1535.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141101_1135.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141102_0820.root'] = (4.50, 5.500, 25)
		
		


	if run == 11:
	
		# Cs calibrations
		#run_files[run]['nerix_141124_1208.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141201_1409.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141203_1116.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141205_1125.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141208_1308.root'] = (4.50, 5.500, -1)

		run_files[run]['nerix_141210_0929.root'] = (4.50, 5.500, -1)
		run_files[run]['nerix_141210_1028.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141212_1527.root'] = (4.50, 5.500, -1)

		run_files[run]['nerix_141215_1625.root'] = (4.50, 5.500, -1)
		run_files[run]['nerix_141216_0925.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141217_0928.root'] = (4.50, 2.356, -1)

		run_files[run]['nerix_141219_1118.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_141219_1816.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_141222_1100.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_150105_1525.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_150106_1100.root'] = (4.50, 0.345, -1)

		run_files[run]['nerix_150106_1137.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_150106_1234.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_150106_1404.root'] = (4.50, 5.500, -1)
		run_files[run]['nerix_150107_1659.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_150112_1213.root'] = (4.50, 1.054, -1)

		run_files[run]['nerix_150114_1224.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_150119_1142.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_150210_1114.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_150213_1200.root'] = (4.50, 1.054, -1)
		
		
		#Co calibrations
		run_files[run]['nerix_141203_1139.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_141208_1209.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_141210_1249.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_141217_1114.root'] = (4.50, 2.356, -2)
		run_files[run]['nerix_141222_1143.root'] = (4.50, 1.054, -2)
		
		run_files[run]['nerix_150114_1541.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150121_1527.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150129_0957.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_150129_1102.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150129_1207.root'] = (4.50, 2.356, -2)
		
		run_files[run]['nerix_150129_1450.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_150204_1313.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150210_1242.root'] = (4.50, 1.054, -2)

		
		# coincidence
		run_files[run]['nerix_141201_1746.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141202_1119.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141202_1757.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141203_1617.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141204_1408.root'] = (4.50, 1.054, 0)

		run_files[run]['nerix_141204_1750.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141209_1810.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141210_1811.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141211_0926.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141212_1716.root'] = (4.50, 5.500, 25)

		run_files[run]['nerix_141213_1236.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141214_1251.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141226_1415.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_141227_1005.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_141228_1253.root'] = (4.50, 5.500, 0)

		run_files[run]['nerix_141230_1307.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_141215_1114.root'] = (4.50, 5.500, 25)
		run_files[run]['nerix_141215_1751.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141216_1037.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141217_1555.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141218_0757.root'] = (4.50, 2.356, 25)

		run_files[run]['nerix_141218_1100.root'] = (4.50, 2.356, 25)
		run_files[run]['nerix_141219_1913.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141220_0956.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141222_1016.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141222_1450.root'] = (4.50, 1.054, 0)

		run_files[run]['nerix_141222_1554.root'] = (4.50, 2.356, 0)
		
		run_files[run]['nerix_150112_1356.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_141222_1601.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_141223_0851.root'] = (4.50, 2.356, 0)
		run_files[run]['nerix_141226_1415.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_141227_1005.root'] = (4.50, 5.500, 0)

		run_files[run]['nerix_141228_1253.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_141230_1307.root'] = (4.50, 5.500, 0)
		run_files[run]['nerix_141230_1801.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_141231_1203.root'] = (4.50, 0.345, 0)
		run_files[run]['nerix_141231_1935.root'] = (4.50, 0.345, 0)

		run_files[run]['nerix_150107_1759.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_150110_1528.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_150112_1356.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_150206_1357.root'] = (4.50, 1.054, 0)
		run_files[run]['nerix_150207_0036.root'] = (4.50, 1.054, 0)
		

	if run == 13:
		run_files[run]['nerix_150702_1441.root'] = (0, 0, -3)
		run_files[run]['nerix_150702_1515.root'] = (0, 0, -3)



	if run == 14:
		run_files[run]['nerix_150819_1223.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150820_1359.root'] = (4.50, 1.054, -3)
		run_files[run]['nerix_150821_1116.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150821_1210.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150821_1455.root'] = (4.50, 1.054, -2)
		
		run_files[run]['nerix_150821_1626.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150824_1226.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_150824_1629.root'] = (4.50, 2.356, -1)
		run_files[run]['nerix_150825_0933.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150825_1008.root'] = (4.50, 1.054, -1)

	if run == 15:
		
		run_files[run]['nerix_150901_1048.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150901_1119.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150901_1153.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150901_1228.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150901_1253.root'] = (4.50, 1.054, -2)

		run_files[run]['nerix_150901_1321.root'] = (4.10, 1.054, -2)
		run_files[run]['nerix_150901_1554.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150902_1029.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150902_1228.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150902_1328.root'] = (4.50, 1.054, -4)

		run_files[run]['nerix_150902_1446.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150902_1512.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150902_1537.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150902_1559.root'] = (4.50, 1.054, -4) #altered for pos rec checks
		run_files[run]['nerix_150902_1620.root'] = (4.50, 1.054, -4)
		
		# pos rec test (all variations of 1559)
		run_files[run]['nerix_150902_1558.root'] = (4.50, 1.054, -4) #orig
		run_files[run]['nerix_150902_1600.root'] = (4.50, 1.054, -4) #r95
		run_files[run]['nerix_150902_1601.root'] = (4.50, 1.054, -4) #r90
		run_files[run]['nerix_150902_1602.root'] = (4.50, 1.054, -4) #r85
		
		run_files[run]['nerix_150902_1643.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150904_1151.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150904_1223.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150904_1254.root'] = (4.50, 5.500, -4)
		run_files[run]['nerix_150908_1325.root'] = (4.50, 5.500, -3)
		
		
		
		run_files[run]['nerix_150908_1330.root'] = (4.50, 5.500, -3)
		run_files[run]['nerix_150908_1334.root'] = (4.50, 5.500, -3)


		run_files[run]['nerix_150910_1822.root'] = (4.50, 1.054, -5) #trig efficiency
		run_files[run]['nerix_150911_1028.root'] = (4.50, 1.054, -3)
		run_files[run]['nerix_150911_1129.root'] = (4.50, 1.054, -3)
		run_files[run]['nerix_150911_1717.root'] = (4.50, 1.054, -5) # trig efficiency
		run_files[run]['nerix_150914_1236.root'] = (4.50, 1.054, -4)

		run_files[run]['nerix_150914_1307.root'] = (3.50, 1.054, -4) # ext efficiency
		run_files[run]['nerix_150914_1346.root'] = (3.70, 1.054, -4) # ext efficiency
		run_files[run]['nerix_150914_1416.root'] = (3.90, 1.054, -4) # ext efficiency
		run_files[run]['nerix_150914_1450.root'] = (4.10, 1.054, -4) # ext efficiency
		run_files[run]['nerix_150914_1519.root'] = (4.30, 1.054, -4) # ext efficiency
		
		run_files[run]['nerix_150914_1549.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_150914_1618.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_150914_1647.root'] = (4.50, 1.500, -4)
		run_files[run]['nerix_150914_1719.root'] = (4.50, 2.356, -4)
		run_files[run]['nerix_150914_1750.root'] = (4.50, 5.500, -4)
		
		run_files[run]['nerix_150915_0946.root'] = (3.70, 1.054, -1) # ext efficiency
		run_files[run]['nerix_150915_1332.root'] = (4.50, 1.054, -4) # ext efficiency
		run_files[run]['nerix_150915_1405.root'] = (4.60, 1.054, -4) # ext efficiency
		run_files[run]['nerix_150915_1435.root'] = (4.70, 1.054, -4) # ext efficiency
		run_files[run]['nerix_150915_1502.root'] = (4.80, 1.054, -4) # ext efficiency

		run_files[run]['nerix_150916_1201.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_150916_1342.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_150916_1559.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_150917_0941.root'] = (4.50, 1.500, -4)
		run_files[run]['nerix_150917_1039.root'] = (4.50, 2.356, -4)
		
		run_files[run]['nerix_150917_1202.root'] = (4.50, 5.500, -4)
		run_files[run]['nerix_150921_1442.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_150928_1150.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_150928_1257.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_150928_1338.root'] = (4.50, 2.356, -2)

		run_files[run]['nerix_150928_1424.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_150928_1516.root'] = (4.50, 0.700, -2)
		run_files[run]['nerix_150928_1622.root'] = (4.50, 1.500, -2)
		run_files[run]['nerix_150928_1717.root'] = (4.50, 1.054, -2) # s2 gain
		run_files[run]['nerix_150929_1719.root'] = (4.50, 2.356, -4)

		run_files[run]['nerix_150930_0945.root'] = (4.50, 5.500, -2)
		run_files[run]['nerix_150930_1043.root'] = (4.50, 5.500, -4) # low gain, Co present
		run_files[run]['nerix_150930_1233.root'] = (4.50, 5.500, -2) # low gain, Co present
		run_files[run]['nerix_150930_1346.root'] = (4.50, 2.356, -2) # low gain, Co present
		run_files[run]['nerix_150930_1456.root'] = (4.50, 1.054, -2) # gain cal, minitron on
		
		run_files[run]['nerix_150930_1604.root'] = (4.50, 0.345, -2) # gain cal, minitron on
		run_files[run]['nerix_150930_1708.root'] = (4.50, 1.500, -2) # gain cal, minitron on
		run_files[run]['nerix_150930_1806.root'] = (4.50, 1.054, -2) # gain cal, minitron on
		run_files[run]['nerix_151001_0916.root'] = (4.50, 1.054, -2) # s2 gain
		run_files[run]['nerix_151001_1236.root'] = (4.50, 1.054, -1) # s2 gain
		
		run_files[run]['nerix_151001_1307.root'] = (4.50, 1.054, -1) # s2 gain
		run_files[run]['nerix_151002_1012.root'] = (4.50, 1.054, -1) # s2 gain
		run_files[run]['nerix_151002_1044.root'] = (4.50, 1.054, -2) # s2 gain
		run_files[run]['nerix_151002_1143.root'] = (4.50, 1.054, -2) # s2 gain
		run_files[run]['nerix_151002_1244.root'] = (4.50, 1.054, -2) # s2 gain

		run_files[run]['nerix_151002_1318.root'] = (4.50, 1.054, -4) # bkg
		run_files[run]['nerix_151002_1517.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_151002_1559.root'] = (4.50, 1.054, -2) # s2 gain
		run_files[run]['nerix_151004_1237.root'] = (4.50, 1.054, -1) # s2 gain
		run_files[run]['nerix_151004_1358.root'] = (4.50, 5.500, -2)
		
		run_files[run]['nerix_151004_1427.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_151004_1458.root'] = (4.50, 1.500, -2)
		run_files[run]['nerix_151004_1525.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_151004_1554.root'] = (4.50, 2.356, -2)
		run_files[run]['nerix_151005_1316.root'] = (4.50, 1.054, -1)

		run_files[run]['nerix_151005_1408.root'] = (4.50, 0.345, -2)
		run_files[run]['nerix_151005_1436.root'] = (4.50, 0.700, -2)
		run_files[run]['nerix_151005_1505.root'] = (4.50, 1.054, -2)
		run_files[run]['nerix_151005_1536.root'] = (4.50, 1.500, -2)
		run_files[run]['nerix_151005_1605.root'] = (4.50, 2.356, -2)

		run_files[run]['nerix_151006_0847.root'] = (4.50, 2.356, -2) #s2 gain
		run_files[run]['nerix_151006_0949.root'] = (4.50, 5.500, -4)
		run_files[run]['nerix_151006_1100.root'] = (4.50, 2.356, -4)
		run_files[run]['nerix_151006_1150.root'] = (4.50, 1.500, -4)
		run_files[run]['nerix_151006_1235.root'] = (4.50, 1.054, -4)

		run_files[run]['nerix_151006_1316.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151006_1404.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151006_1454.root'] = (3.70, 1.054, -4)
		run_files[run]['nerix_151006_1522.root'] = (3.90, 1.054, -4)
		run_files[run]['nerix_151006_1547.root'] = (4.10, 1.054, -4)

		run_files[run]['nerix_151006_1612.root'] = (4.30, 1.054, -4)
		run_files[run]['nerix_151006_1637.root'] = (4.70, 1.054, -4)
		run_files[run]['nerix_151006_1703.root'] = (4.90, 1.054, -4)
		run_files[run]['nerix_151007_0931.root'] = (3.70, 1.054, -1)
		run_files[run]['nerix_151008_1005.root'] = (3.90, 1.054, -1)
		
		run_files[run]['nerix_151008_1254.root'] = (4.10, 1.054, -1)
		run_files[run]['nerix_151009_0924.root'] = (4.30, 1.054, -1)
		run_files[run]['nerix_151009_1026.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_151009_1133.root'] = (4.70, 1.054, -1)
		run_files[run]['nerix_151009_1235.root'] = (4.90, 1.054, -1)

		run_files[run]['nerix_151011_1136.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151011_1212.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151011_1249.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151011_1328.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151011_1405.root'] = (4.50, 1.054, -5) # tac cal
		
		run_files[run]['nerix_151011_1446.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151011_1526.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151011_1609.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151013_1409.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151013_1440.root'] = (4.50, 1.054, -5) # tac cal

		run_files[run]['nerix_151013_1718.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151013_1738.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151013_1746.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151014_1201.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151014_1233.root'] = (4.50, 1.054, -5) # tac cal

		run_files[run]['nerix_151014_1606.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151014_1635.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151014_1717.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151014_1734.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151015_1044.root'] = (4.50, 1.054, -1) # collimated

		run_files[run]['nerix_151015_1209.root'] = (4.50, 1.054, -1) # collimated
		run_files[run]['nerix_151015_1228.root'] = (4.50, 1.054, -1) # collimated
		run_files[run]['nerix_151015_1331.root'] = (4.50, 1.054, -1) # collimated
		run_files[run]['nerix_151015_1406.root'] = (4.50, 1.054, -1) # collimated
		run_files[run]['nerix_151015_1530.root'] = (4.50, 1.054, -5) # tac cal

		run_files[run]['nerix_151015_1540.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151015_1551.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151015_1607.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151015_1713.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151016_0731.root'] = (4.50, 1.054, -5) # tac cal

		run_files[run]['nerix_151016_0941.root'] = (4.50, 1.054, -5) # tac cal
		run_files[run]['nerix_151016_1003.root'] = (4.50, 1.054, -5)
		run_files[run]['nerix_151019_1426.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151019_1502.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151019_1546.root'] = (4.50, 1.054, -4)

		run_files[run]['nerix_151019_1635.root'] = (4.50, 2.356, -4)
		run_files[run]['nerix_151020_1104.root'] = (4.50, 1.054, -4) #tac efficiency
		run_files[run]['nerix_151026_1215.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151026_1253.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151026_1342.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151026_1421.root'] = (4.50, 1.500, -4)
		
		run_files[run]['nerix_151026_1506.root'] = (4.50, 2.356, -4)
		run_files[run]['nerix_151026_1548.root'] = (4.50, 1.054, -4) # gas gain
		
		run_files[run]['nerix_151028_1057.root'] = (3.00, 1.054, -4) # gas gain
		run_files[run]['nerix_151028_1140.root'] = (3.00, 0.345, -4)
		run_files[run]['nerix_151028_1205.root'] = (3.00, 1.054, -4)
		run_files[run]['nerix_151028_1231.root'] = (3.00, 1.500, -4)
		run_files[run]['nerix_151028_1300.root'] = (3.00, 2.356, -4)

		run_files[run]['nerix_151029_0923.root'] = (4.50, 1.054, -4) #gas gain
		run_files[run]['nerix_151029_1026.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151029_1143.root'] = (4.50, 1.054, -4) #gas gain
		run_files[run]['nerix_151029_1240.root'] = (4.50, 1.054, -4) #gas gain
		run_files[run]['nerix_151102_1033.root'] = (4.50, 1.054, -1) #gas gain
		
		run_files[run]['nerix_151102_1126.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151102_1154.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151102_1232.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151102_1307.root'] = (4.50, 1.500, -4)
		run_files[run]['nerix_151102_1355.root'] = (4.50, 2.356, -4)
		
		run_files[run]['nerix_151102_1529.root'] = (4.50, 1.054, -4) # bkg
		run_files[run]['nerix_151102_1713.root'] = (4.50, 1.054, -4) # bkg
		run_files[run]['nerix_151103_0915.root'] = (4.50, 0.345, -4) # bkg
		run_files[run]['nerix_151103_1253.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151106_1333.root'] = (4.50, 1.054, -4)
		
		run_files[run]['nerix_151106_1545.root'] = (4.50, 1.054, -4) #EJ3
		run_files[run]['nerix_151106_1607.root'] = (4.50, 1.054, -4) #EJ3,4
		run_files[run]['nerix_151106_1617.root'] = (4.50, 1.054, -4) #EJ3,4,5
		run_files[run]['nerix_151106_1624.root'] = (4.50, 1.054, -4) #EJ2,3,4,5
		run_files[run]['nerix_151109_1527.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151109_1646.root'] = (4.50, 1.054, -1) #gas gain
		
		run_files[run]['nerix_151109_1727.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151109_1952.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151109_2204.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151110_0213.root'] = (4.50, 1.500, -4)
		run_files[run]['nerix_151110_0650.root'] = (4.50, 2.356, -4)
		
		run_files[run]['nerix_151110_1701.root'] = (4.50, 1.054, -1) # gas gain
		run_files[run]['nerix_151112_1001.root'] = (4.50, 1.054, -1) # gas gain
		run_files[run]['nerix_151116_0937.root'] = (4.50, 1.054, -1) # gas gain
		run_files[run]['nerix_151116_1022.root'] = (4.50, 1.054, -1) # gas gain
		run_files[run]['nerix_151116_1552.root'] = (4.50, 1.054, -4)
		
		run_files[run]['nerix_151117_1237.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151117_1307.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151117_1336.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151117_1403.root'] = (4.50, 1.500, -4)
		run_files[run]['nerix_151117_1445.root'] = (4.50, 2.356, -4)
		
		run_files[run]['nerix_151119_0925.root'] = (4.50, 0.345, -6)
		run_files[run]['nerix_151119_1016.root'] = (4.50, 0.700, -6)
		run_files[run]['nerix_151119_1117.root'] = (4.50, 1.054, -6)
		run_files[run]['nerix_151119_1210.root'] = (4.50, 1.500, -6)
		run_files[run]['nerix_151119_1308.root'] = (4.50, 2.356, -6)

		run_files[run]['nerix_151119_1409.root'] = (4.50, 2.356, -1) # PMT5=610V (up from 580V)
		run_files[run]['nerix_151119_1447.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_151119_1506.root'] = (4.50, 0.700, -1)
		run_files[run]['nerix_151119_1527.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_151119_1546.root'] = (4.50, 1.500, -1)
		run_files[run]['nerix_151119_1628.root'] = (4.50, 2.356, -1)
		
		run_files[run]['nerix_151120_1056.root'] = (4.50, 1.054, -1) # gas gain
		run_files[run]['nerix_151123_1214.root'] = (4.50, 1.054, -4)
		
		run_files[run]['nerix_151123_1309.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151123_1347.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151123_1417.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151123_1445.root'] = (4.50, 1.500, -4)
		run_files[run]['nerix_151123_1522.root'] = (4.50, 2.356, -4)
		
		run_files[run]['nerix_151124_1719.root'] = (4.50, 0.345, -6)
		run_files[run]['nerix_151124_2014.root'] = (4.50, 0.700, -6)
		run_files[run]['nerix_151124_2303.root'] = (4.50, 1.054, -6)
		run_files[run]['nerix_151125_0625.root'] = (4.50, 1.500, -6)
		run_files[run]['nerix_151125_0929.root'] = (4.50, 2.356, -6)
		
		run_files[run]['nerix_151201_1232.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_151201_1249.root'] = (4.50, 0.700, -1)
		run_files[run]['nerix_151201_1306.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_151201_1327.root'] = (4.50, 1.500, -1)
		run_files[run]['nerix_151201_1344.root'] = (4.50, 2.356, -1)
		
		run_files[run]['nerix_151201_1430.root'] = (4.50, 2.356, -4)
		run_files[run]['nerix_151201_1519.root'] = (4.50, 0.345, -4)
		#run_files[run]['nerix_151201_1520.root'] = (4.50, 0.345, -4)
		#run_files[run]['nerix_151201_1600.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151201_1639.root'] = (4.50, 0.700, -4)
		run_files[run]['nerix_151201_1722.root'] = (4.50, 1.054, -4)
		run_files[run]['nerix_151202_1.root'] = (4.50, 1.500, -4)
		
		run_files[run]['nerix_151203_1119.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_151203_1138.root'] = (4.50, 0.700, -1)
		run_files[run]['nerix_151203_1156.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_151203_1221.root'] = (4.50, 1.500, -1)
		run_files[run]['nerix_151203_1239.root'] = (4.50, 2.356, -1)
		
		run_files[run]['nerix_151203_1317.root'] = (4.50, 0.345, -5)
		run_files[run]['nerix_151203_1345.root'] = (4.50, 0.700, -5)
		run_files[run]['nerix_151203_1411.root'] = (4.50, 1.054, -5)
		run_files[run]['nerix_151203_1438.root'] = (4.50, 1.500, -5)
		run_files[run]['nerix_151203_1549.root'] = (4.50, 2.356, -5)
		
		run_files[run]['nerix_151207_1342.root'] = (4.50, 2.356, -1) # gas gain
		
		run_files[run]['nerix_151207_1427.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_151207_1514.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_151207_1536.root'] = (4.50, 0.700, -1)
		run_files[run]['nerix_151207_1557.root'] = (4.50, 1.500, -1)
		run_files[run]['nerix_151207_1619.root'] = (4.50, 2.356, -1)
		
		run_files[run]['nerix_151208_0934.root'] = (4.50, 2.356, -4)
		run_files[run]['nerix_151208_1057.root'] = (4.50, 0.345, -4)
		run_files[run]['nerix_151208_1400.root'] = (4.50, 2.356, -4) #V5=610V
		run_files[run]['nerix_151209_1008.root'] = (4.50, 0.345, -4) #V5=610V
		run_files[run]['nerix_151209_1126.root'] = (4.50, 0.700, -4) #V5=610V
		
		run_files[run]['nerix_151209_1240.root'] = (4.50, 1.054, -4) #V5=610V
		run_files[run]['nerix_151209_1353.root'] = (4.50, 1.500, -4) #V5=610V
		run_files[run]['nerix_151210_0930.root'] = (4.50, 0.345, -1) #V5=530V
		run_files[run]['nerix_151210_1130.root'] = (4.50, 0.345, -1) #V5=515V
		run_files[run]['nerix_151210_1237.root'] = (4.50, 0.345, -1) #V5=500V

		run_files[run]['nerix_151210_1341.root'] = (4.50, 0.345, -1) #V5=545V
		run_files[run]['nerix_151210_1437.root'] = (4.50, 0.345, -1) #V5=560V
		run_files[run]['nerix_151210_1531.root'] = (4.50, 0.345, -1) #V5=575V
		run_files[run]['nerix_151210_1646.root'] = (4.50, 0.345, -1) #V5=590V
		run_files[run]['nerix_151211_0926.root'] = (4.50, 0.345, -1) #V5=485V
		
		run_files[run]['nerix_151211_1037.root'] = (4.50, 0.345, -1) #V5=470V
		run_files[run]['nerix_151211_1219.root'] = (4.50, 0.345, -1) #V5=455V (can't see peak)
		
		run_files[run]['nerix_151214_1126.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_151214_1213.root'] = (4.50, 0.700, -1)
		run_files[run]['nerix_151214_1244.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_151214_1313.root'] = (4.50, 1.500, -1)
		run_files[run]['nerix_151214_1344.root'] = (4.50, 2.356, -1)
		
		run_files[run]['nerix_151215_1059.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_151215_1524.root'] = (4.50, 0.345, -1) # HV on
		run_files[run]['nerix_151216_1018.root'] = (4.50, 0.345, -6)
		run_files[run]['nerix_151216_1139.root'] = (4.50, 1.054, -6)
		run_files[run]['nerix_151216_1251.root'] = (4.50, 2.356, -6)
		
		run_files[run]['nerix_151216_1419.root'] = (4.50, 0.700, -6)
		run_files[run]['nerix_151216_1532.root'] = (4.50, 1.500, -6)
		run_files[run]['nerix_151217_1114.root'] = (4.50, 0.345, -1) # 90deg from norm
		run_files[run]['nerix_151217_1304.root'] = (4.50, 0.345, -1) # 180deg from norm
		run_files[run]['nerix_151217_1437.root'] = (4.50, 0.345, -1) # 270deg from norm
		
		run_files[run]['nerix_151218_1227.root'] = (4.50, 0.345, -4)
		
		run_files[run]['nerix_151222_1200.root'] = (4.50, 0.345, -1)
		run_files[run]['nerix_151222_1227.root'] = (4.50, 0.700, -1)
		run_files[run]['nerix_151222_1253.root'] = (4.50, 1.054, -1)
		run_files[run]['nerix_151222_1320.root'] = (4.50, 1.500, -1)
		run_files[run]['nerix_151222_1349.root'] = (4.50, 2.356, -1)
		
		
		# coincidence


		run_files[run]['nerix_151022_1753.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151023_0830.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		#run_files[run]['nerix_151027_1609.root'] = (4.50, 1.054, 45) # TAC off
		#run_files[run]['nerix_151027_1703.root'] = (4.50, 1.054, 45) # TAC off
		run_files[run]['nerix_151028_1412.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})

		run_files[run]['nerix_151028_1659.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151029_1428.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151029_1753.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151030_1036.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
		run_files[run]['nerix_151031_1156.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
		
		run_files[run]['nerix_151101_1124.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
		run_files[run]['nerix_151102_0728.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
		#run_files[run]['nerix_151103_1340.root'] = (4.50, 1.054, 45) # high rate - bad
		#run_files[run]['nerix_151103_1643.root'] = (4.50, 1.054, 45) # high rate - bad
		#run_files[run]['nerix_151104_1038.root'] = (4.50, 1.054, 45) # high rate - bad
		
		#run_files[run]['nerix_151104_1212.root'] = (4.50, 1.054, 45) # high rate - bad
		#run_files[run]['nerix_151104_1348.root'] = (4.50, 1.054, 45) # high rate - bad
		run_files[run]['nerix_151104_1654.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151105_0922.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151105_1559.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		
		run_files[run]['nerix_151106_1223.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151106_1627.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151108_0727.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		#run_files[run]['nerix_151110_1054.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # higher rate - bad
		#run_files[run]['nerix_151110_1547.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # higher rate - bad
		
		run_files[run]['nerix_151111_0951.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151111_1601.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		#run_files[run]['nerix_151112_1045.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # higher rate - bad
		#run_files[run]['nerix_151112_1446.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # rate issue
		run_files[run]['nerix_151113_0921.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		
		run_files[run]['nerix_151113_1717.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151115_1236.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		#run_files[run]['nerix_151116_1640.root'] = (4.50, 1.054, {45:[0,1], 37:[2,3]}) # baseline shift on EJs
		run_files[run]['nerix_151117_1537.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151119_1651.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		
		run_files[run]['nerix_151120_1137.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151120_1649.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151122_2120.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151123_1559.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		
		#run_files[run]['nerix_151125_1227.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151125_1407.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151125_1823.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151128_1116.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151129_1927.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		
		run_files[run]['nerix_151201_1809.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151202_1039.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151203_1643.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151204_1407.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151206_2106.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		
		run_files[run]['nerix_151207_1712.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
		
		run_files[run]['nerix_151208_1612.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151209_1508.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151210_1835.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151211_1500.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151212_0911.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})

		run_files[run]['nerix_151214_1435.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151215_1754.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151216_1730.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151217_1632.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151218_1320.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		
		#run_files[run]['nerix_151218_1439.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]}) #PMT3 off for time
		run_files[run]['nerix_151220_1308.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151222_1515.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151222_1546.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
		
		
		# separate for comparisons
		"""
		run_files[run]['nerix_151029_1753.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151030_1036.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
		run_files[run]['nerix_151031_1156.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
		run_files[run]['nerix_151106_1627.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		run_files[run]['nerix_151113_1717.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
		"""

