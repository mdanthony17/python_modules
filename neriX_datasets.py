run_files = {}

PARAMETERS_INDEX = 0
# once inside voltage Index
ANODE_INDEX = 0
CATHODE_INDEX = 1
DEGREE_INDEX = 2 #-10 for gas gain, -7 for EJ cal, -6 for bkg, -5 for na, -4 minitron, -3 for TDC, -2 (for co cal), -1 (for cs cal), 0 deg, or 25 deg


dTOFBounds = {(45, 1.054):(5, 40),#(25, 50),
               (30, 1.054):(10, 50),#(45, 80),
               (45, 2.356):(5, 40),
               (30, 2.356):(10, 50),
               (45, 0.345):(5, 40),
               (30, 0.345):(10, 50),
               (62, 1.054):(-5, 15),
               (35, 1.054):(0, 20),
               (62, 2.356):(-5, 15),
               (35, 2.356):(0, 20),
               (62, 0.345):(-5, 15),
               (62, 0.345):(-5, 15),
               (35, 0.345):(0, 20)
              }

lTOFAccidentalsRanges = [[-50, -35], [80, 100]]

lLiqSciS1DtRange = [6, 16]
lLiqSciS1DtAccidentalsRanges = [[-80, -40], [40, 60]]

# set runs to use here
runsInUse = [10, 11, 13, 14, 15, 16]

d_dt_offset_gate = {10:0, 11:0, 15:1.44, 16:1.44}

for run in runsInUse:
    run_files[run] = {}

for run in run_files:
    if run == 10:
        # run_10

        #Cs calibration files
        run_files[run]['nerix_140915_1043.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_140915_1158.root'] = (4.50, 2.356, -1)
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
        run_files[run]['nerix_141029_1110.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_141103_1101.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_141103_1119.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_141103_1553.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_141103_1627.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_141103_1655.root'] = (4.50, 0.345, -1)


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
        run_files[run]['nerix_140928_2321.root'] = (4.50, 5.500, 0)
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
        run_files[run]['nerix_141203_1506.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_141205_1125.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_141208_1123.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_141208_1308.root'] = (4.50, 5.500, -1)

        run_files[run]['nerix_141210_0929.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_141210_1028.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_141212_1527.root'] = (4.50, 5.500, -1)

        run_files[run]['nerix_141215_1625.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_141216_0925.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_141217_0928.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_141217_1041.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_141219_1118.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_141219_1816.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_141222_1100.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_141222_1244.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150105_1525.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_150106_1100.root'] = (4.50, 0.345, -1)

        run_files[run]['nerix_150106_1137.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150106_1234.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_150106_1404.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_150107_1659.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150112_1213.root'] = (4.50, 1.054, -1)

        run_files[run]['nerix_150114_1224.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150114_1407.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150128_1158.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_150128_1327.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150128_1454.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_150128_1622.root'] = (4.50, 5.500, -1)
        run_files[run]['nerix_150119_1142.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150204_1210.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150210_1114.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_150210_1210.root'] = (4.50, 1.054, -1)

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

        run_files[run]['nerix_151222_1431.root'] = (4.50, 0.345, -4)

        run_files[run]['nerix_151222_1200.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_151222_1227.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_151222_1253.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_151222_1320.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_151222_1349.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_151223_1157.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_151223_1222.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_151223_1249.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_151223_1317.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_151223_1343.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_151223_1411.root'] = (4.50, 2.356, -6) #bad
        run_files[run]['nerix_151223_1537.root'] = (4.50, 1.054, -6) #bad
        run_files[run]['nerix_151223_1700.root'] = (4.50, 0.345, -6) #bad

        run_files[run]['nerix_151228_1052.root'] = (4.50, 0.345, -4)


        # below are in /home/xedaq

        run_files[run]['nerix_151228_1318.root'] = (4.50, 0.345, -1) # gas gain

        run_files[run]['nerix_151228_1409.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_151228_1433.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_151228_1454.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_151228_1544.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_151228_1607.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_151228_1638.root'] = (4.50, 0.345, -6) #bad
        run_files[run]['nerix_151228_1843.root'] = (4.50, 0.700, -6) #bad
        run_files[run]['nerix_151228_2035.root'] = (4.50, 1.054, -6) #bad
        run_files[run]['nerix_151228_2229.root'] = (4.50, 1.500, -6) #bad
        run_files[run]['nerix_151229_0741.root'] = (4.50, 2.356, -6) #bad

        run_files[run]['nerix_151229_1029.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_151229_1054.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_151229_1120.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_151229_1143.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_151229_1206.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160104_1113.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160104_1140.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160104_1208.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160104_1236.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160104_1303.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160104_1342.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160104_1410.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160104_1436.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160104_1502.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160104_1528.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160104_1708.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160104_1840.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160104_2013.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160104_2200.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160105_0713.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160104_1558.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_160105_0939.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160111_0935.root'] = (4.50, 2.356, -4)
        run_files[run]['nerix_160111_1039.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160111_1145.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160111_1213.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160111_1245.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160111_1306.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160111_1324.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160112_1250.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160112_1326.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160112_1357.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160112_1449.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160112_1518.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160112_1551.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160112_1701.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160112_1919.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160112_2151.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160113_0628.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160118_0922.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160118_1017.root'] = (4.50, 1.054, -1)

        run_files[run]['nerix_160118_1120.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160118_1147.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160118_1214.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160118_1241.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160118_1309.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160118_1341.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160118_1409.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160118_1436.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160118_1502.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160118_1529.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160118_1558.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160118_1737.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160118_1853.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160118_2007.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160118_2209.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160125_1023.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160125_1125.root'] = (4.50, 1.054, -1)

        run_files[run]['nerix_160125_1232.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160125_1300.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160125_1327.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160125_1353.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160125_1423.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160125_1452.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160125_1521.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160125_1550.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160125_1618.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160125_1645.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160126_0946.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160126_1055.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160126_1201.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160126_1309.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160126_1418.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160127_1300.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160127_1352.root'] = (4.50, 1.054, -1)

        run_files[run]['nerix_160127_1502.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160127_1530.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160127_1556.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160127_1623.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160127_1649.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160128_1513.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160203_0931.root'] = (4.50, 1.054, -1)


        run_files[run]['nerix_160201_2000.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160201_2312.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160202_0626.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160202_0914.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160202_1037.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160204_0922.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160204_1026.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160204_1053.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160204_1119.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160204_1145.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160204_1212.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160204_1242.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160204_1310.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160204_1337.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160204_1404.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160204_1438.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160208_0941.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160208_1010.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160208_1039.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160208_1109.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160208_1137.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160208_1334.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160208_1402.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160208_1428.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160208_1454.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160208_1521.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160208_1709.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160208_1817.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160208_1928.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160208_2038.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160208_2205.root'] = (4.50, 2.356, -6)


        run_files[run]['nerix_160209_0955.root'] = (4.50, 1.054, -4)

        # need to analyze below

        run_files[run]['nerix_160216_0925.root'] = (4.50, 0.345, -4)

        run_files[run]['nerix_160216_1305.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160216_1332.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160216_1403.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160216_1431.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160216_1458.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160216_1530.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160216_1557.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160216_1623.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160216_1651.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160216_1718.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160216_1749.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160216_2012.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160216_2132.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160216_2255.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160217_0643.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160219_1248.root'] = (4.50, 2.356, -1)
        run_files[run]['nerix_160219_1433.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160222_1048.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160222_1157.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160222_1224.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160222_1254.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160222_1325.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160222_1353.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160222_1449.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160222_1518.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160222_1545.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160222_1613.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160222_1640.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160222_1709.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160222_1851.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160222_2200.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160222_2310.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160223_0627.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160223_0947.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160229_1015.root'] = (4.50, 2.356, -4)
        run_files[run]['nerix_160229_1116.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160229_1223.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160229_1250.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160229_1316.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160229_1342.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160229_1408.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160229_1435.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160229_1504.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160229_1534.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160229_1605.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160229_1631.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160229_1659.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160229_1814.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160229_2237.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160301_0725.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160301_0914.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160301_1130.root'] = (4.50, 2.356, -7)







        # coincidence

        # EJs further

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
        """

        """
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
        run_files[run]['nerix_151223_1850.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_151224_1911.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_151225_2037.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_151227_1109.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})


        # ***************
        # 151228_1318 until ______ are in /home/xedaq/run_15/ instead of normal location

        run_files[run]['nerix_151229_1316.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_151231_0844.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160101_1034.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160102_1646.root'] = (4.50, 0.345, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160103_1226.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160105_1040.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160106_0916.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160107_1043.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160108_1413.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160111_1403.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160113_1015.root'] = (4.50, 2.356, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160113_1230.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160114_1602.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160115_0913.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160117_1244.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160119_1125.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160120_1134.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160121_1554.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160122_0926.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160124_1554.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160125_1737.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160126_1706.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160127_1747.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160128_0922.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160128_1741.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160129_0924.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160130_1026.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_160131_1031.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})

        run_files[run]['nerix_160202_1316.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160202_1623.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160203_1620.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160204_1729.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160205_0804.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})

        run_files[run]['nerix_160206_1046.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160209_1058.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160209_1545.root'] = (4.50, 1.054, {62:[2,3], 35:[0,1]})

        run_files[run]['nerix_160210_1032.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160210_1547.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160211_0924.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160211_1030.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160211_1410.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})

        run_files[run]['nerix_160212_0917.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160213_1219.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160215_0942.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160217_0944.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160218_0921.root'] = (4.50, 0.345, {62:[2,3], 35:[0,1]})

        run_files[run]['nerix_160218_1158.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160218_1525.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160219_1529.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160219_1715.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160221_1049.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})

        run_files[run]['nerix_160224_1542.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160226_1014.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160226_1455.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160226_1610.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160228_1801.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})


        # ***************

        """
        run_files[run]['nerix_160301_1726.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160302_0840.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160302_1401.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160302_1659.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160303_0938.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160307_1154.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        """




        # TRIAL with EJ in place of TAC in trigger
        """
        run_files[run]['nerix_160223_1039.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160223_1043.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160223_1246.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160223_1430.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        run_files[run]['nerix_160224_1039.root'] = (4.50, 2.356, {62:[2,3], 35:[0,1]})
        """

        # separate for comparisons (high rate)
        """
        run_files[run]['nerix_151029_1753.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_151030_1036.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
        run_files[run]['nerix_151031_1156.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]}) # led on
        run_files[run]['nerix_151106_1627.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        run_files[run]['nerix_151113_1717.root'] = (4.50, 1.054, {45:[0,1], 30:[2,3]})
        """


    if run == 16:

        # change of KNF pump and getter
        """
        run_files[run]['nerix_160316_0947.root'] = (4.50, 1.054, -1) # flow~2 SLPM
        run_files[run]['nerix_160316_1308.root'] = (4.50, 1.054, -1) # flow~2.8 SLPM
        run_files[run]['nerix_160316_1401.root'] = (4.50, 1.054, -1) # gas gain

        run_files[run]['nerix_160319_1022.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160319_1424.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160318_1446.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160319_1913.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160319_2335.root'] = (4.50, 2.356, -1)


        run_files[run]['nerix_160321_1033.root'] = (4.50, 1.054, -10) # gas gain
        run_files[run]['nerix_160321_1231.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160321_1416.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160321_1639.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160321_2224.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160322_1124.root'] = (4.50, 1.054, -1) # feedthrough at .175"
        run_files[run]['nerix_160322_1218.root'] = (4.50, 1.054, -1) # feedthrough at .190"
        run_files[run]['nerix_160322_1249.root'] = (4.50, 1.054, -10) # gas gain
        run_files[run]['nerix_160322_1435.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160323_1019.root'] = (4.50, 1.054, -10) # gas gain

        run_files[run]['nerix_160324_0921.root'] = (4.50, 1.054, -10) # gas gain
        """

        # need to analyze below

        # very difficult to fit (minitron off long time)
        """
        run_files[run]['nerix_160322_1712.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160322_2005.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160322_2117.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160323_0652.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160323_0843.root'] = (4.50, 2.356, -6)
        """

        """
        run_files[run]['nerix_160323_1019.root'] = (4.50, 1.054, -10) # gas gain

        run_files[run]['nerix_160323_1323.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160323_1358.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160323_1433.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160323_1508.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160323_1539.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160324_0921.root'] = (4.50, 1.054, -10) # gas gain
        run_files[run]['nerix_160328_0654.root'] = (4.50, 1.054, -10) # gas gain

        run_files[run]['nerix_160328_0918.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160328_1022.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160328_1053.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160328_1125.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160328_1156.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160328_1228.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160328_1302.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160328_1342.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160328_1413.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160328_1443.root'] = (4.50, 2.356, -5)

        # very difficult to fit (minitron off long time)
        run_files[run]['nerix_160328_1629.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160328_1738.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160328_1921.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160329_0000.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160329_0709.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160329_0947.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160329_1631.root'] = (4.50, 1.054, -1) #f-t=.175
        run_files[run]['nerix_160329_1712.root'] = (4.50, 1.054, -1) #f-t=.200
        run_files[run]['nerix_160329_1751.root'] = (4.50, 1.054, -1) #f-t=.225
        run_files[run]['nerix_160329_1825.root'] = (4.50, 1.054, -1) #f-t=.250

        run_files[run]['nerix_160329_2014.root'] = (4.50, 0.345, -1) #f-t=.250
        run_files[run]['nerix_160329_2050.root'] = (4.50, 0.700, -1) #f-t=.250
        run_files[run]['nerix_160329_2117.root'] = (4.50, 1.054, -1) #f-t=.250
        run_files[run]['nerix_160329_2149.root'] = (4.50, 1.500, -1) #f-t=.250
        run_files[run]['nerix_160329_2215.root'] = (4.50, 2.356, -1) #f-t=.250
        """

        # FEED THROUGH FINALED AT 0.25"

        run_files[run]['nerix_160329_2245.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250

        # uniformity, f-t:0.25"
        run_files[run]['nerix_160330_0656.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160330_0918.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160330_1037.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160330_1159.root'] = (4.50, 0.345, -1)

        run_files[run]['nerix_160330_1446.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160330_1637.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160331_1031.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160331_1104.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160331_1131.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160331_1205.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160331_1232.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160331_1549.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160404_1059.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160404_1204.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160404_1232.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160404_1259.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160404_1325.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160404_1350.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160404_1421.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160404_1447.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160404_1530.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160404_1555.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160404_1621.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160404_1649.root'] = (4.50, 1.054, -2) # leveling

        run_files[run]['nerix_160404_1802.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160405_0034.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160405_0625.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160405_0737.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160405_0915.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160405_1054.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160407_1357.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250

        #run_files[run]['nerix_160411_0953.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160411_0612.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160411_0644.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160411_0712.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160411_0739.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160411_0925.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160411_1245.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160411_1313.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160411_1341.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160411_1419.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160411_1447.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160411_1607.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160411_1718.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160411_2149.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160412_0625.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160412_0737.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160412_0919.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250
        run_files[run]['nerix_160412_1052.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160414_1215.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160418_0919.root'] = (4.50, 0.345, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160418_1026.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160418_1052.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160418_1120.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160418_1145.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160418_1210.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160418_1239.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160418_1307.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160418_1337.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160418_1405.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160418_1434.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160418_1621.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160418_1824.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160418_1944.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160418_2127.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160419_0649.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160419_0944.root'] = (4.50, 0.345, -4)
        run_files[run]['nerix_160419_1331.root'] = (4.50, 0.345, -4)

        run_files[run]['nerix_160421_1356.root'] = (4.50, 0.345, -10)

        run_files[run]['nerix_160425_0918.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160425_1206.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160425_1234.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160425_1327.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160425_1355.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160425_1442.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160427_1518.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160428_1124.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160429_1124.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250
        run_files[run]['nerix_160429_1332.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160502_0940.root'] = (4.50, 0.345, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160502_1059.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160502_1134.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160502_1208.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160502_1234.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160502_1302.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160502_1334.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160502_1404.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160502_1433.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160502_1537.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160502_1605.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160502_1742.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160502_1855.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160502_2033.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160502_2109.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160502_2231.root'] = (4.50, 2.356, -6)

        #run_files[run]['nerix_160503_0945.root'] = (4.50, 0.345, -4) # bad -no NR band

        run_files[run]['nerix_160505_0935.root'] = (4.50, 0.345, -10)
        run_files[run]['nerix_160505_1709.root'] = (4.50, 0.345, -4)

        run_files[run]['nerix_160509_1050.root'] = (4.50, 0.345, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160509_1157.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160509_1226.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160509_1252.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160509_1320.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160509_1350.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160509_1419.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160509_1448.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160509_1515.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160509_1544.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160509_1619.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160509_1657.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160509_1923.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160509_2133.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160510_0625.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160510_0753.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160512_1224.root'] = (4.50, 0.345, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160512_1531.root'] = (4.50, 0.345, -4)

        run_files[run]['nerix_160516_0958.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160516_1226.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160516_1255.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160516_1322.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160516_1351.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160516_1418.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160516_1447.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160516_1516.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160516_1545.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160516_1613.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160516_1641.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160516_1715.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160516_1850.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160516_2019.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160516_2247.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160517_0611.root'] = (4.50, 2.356, -6)

        #run_files[run]['nerix_160517_0923.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160517_1012.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160517_1037.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160517_1103.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160517_1133.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160517_1158.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160517_1413.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160519_1135.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160520_1359.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160520_1433.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160520_1615.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160520_1714.root'] = (4.50, 1.054, -4)
        run_files[run]['nerix_160520_1755.root'] = (4.50, 0.345, -1)

        #run_files[run]['nerix_160523_1106.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250 (BAD)

        run_files[run]['nerix_160523_1215.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160523_1242.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160523_1308.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160523_1337.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160523_1405.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160523_1436.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160523_1506.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160523_1534.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160523_1600.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160523_1628.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160523_1700.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160523_2109.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160523_2228.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160524_0623.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160524_0746.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160524_1140.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250
        run_files[run]['nerix_160524_1359.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160525_1500.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160531_1031.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160531_1144.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160531_1234.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160531_1303.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160531_1331.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160531_1357.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160531_1428.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160531_1457.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160531_1525.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160531_1553.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160531_1621.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160531_1654.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160531_1841.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160531_2039.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160531_2200.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160531_2321.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160601_0936.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160601_1001.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160601_1025.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160601_1048.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160601_1116.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160604_1321.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160606_1046.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160606_1155.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160606_1225.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160606_1254.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160606_1347.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160606_1421.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160606_1451.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160606_1521.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160606_1549.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160606_1617.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160606_1646.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160606_1739.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160606_1954.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160606_2211.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160606_0626.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160606_0747.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160607_0920.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160607_0947.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160607_1014.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160607_1041.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160607_1111.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160607_1204.root'] = (4.50, 2.356, -4)
        
        run_files[run]['nerix_160609_1106.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160609_1133.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160609_1200.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160609_1227.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160609_1254.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160610_1726.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160613_0914.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160613_1647.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160614_1054.root'] = (4.50, 0.345, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160614_1203.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160614_1231.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160614_1258.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160614_1328.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160614_1356.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160614_1426.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160614_1455.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160614_1523.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160614_1555.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160614_1623.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160614_1655.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160614_2158.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160614_2321.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160615_0625.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160615_0748.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160615_0935.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160615_1001.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160615_1028.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160615_1056.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160615_1148.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160615_1246.root'] = (4.50, 0.345, -4)

        run_files[run]['nerix_160620_1048.root'] = (4.50, 1.054, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160620_1158.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160620_1226.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160620_1253.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160620_1322.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160620_1349.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160620_1419.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160620_1452.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160620_1522.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160620_1551.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160620_1621.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160620_1653.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160620_2221.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160620_2346.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160621_0621.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160621_0740.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160621_0917.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160621_0944.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160621_1012.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160621_1039.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160621_1107.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160621_1203.root'] = (4.50, 1.054, -4)

        run_files[run]['nerix_160623_1213.root'] = (4.50, 1.054, -4)


        run_files[run]['nerix_160627_1046.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160627_1156.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160627_1224.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160627_1253.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160627_1321.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160627_1349.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160627_1417.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160627_1448.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160627_1531.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160627_1601.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160627_1630.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160627_1701.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160627_1840.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160627_2109.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160628_0615.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160628_0736.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160628_0907.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160628_0959.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160628_1024.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160628_1050.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160628_1116.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160628_1201.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160630_1053.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160705_1049.root'] = (4.50, 2.356, -10) #gas gain,f-t=.250

        run_files[run]['nerix_160705_1205.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160705_1232.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160705_1300.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160705_1325.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160705_1354.root'] = (4.50, 2.356, -1)

        run_files[run]['nerix_160705_1423.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160705_1454.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160705_1523.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160705_1553.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160705_1623.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160705_1655.root'] = (4.50, 0.345, -6)
        run_files[run]['nerix_160705_1926.root'] = (4.50, 0.700, -6)
        run_files[run]['nerix_160705_2100.root'] = (4.50, 1.054, -6)
        run_files[run]['nerix_160705_2222.root'] = (4.50, 1.500, -6)
        run_files[run]['nerix_160706_0522.root'] = (4.50, 2.356, -6)

        run_files[run]['nerix_160706_0824.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160706_0853.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160706_0925.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160706_0954.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160706_1023.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160706_1117.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160711_0911.root'] = (4.50, 0.345, -10)
        
        run_files[run]['nerix_160711_1021.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160711_1050.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160711_1120.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160711_1150.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160711_1219.root'] = (4.50, 2.356, -1)
        
        run_files[run]['nerix_160711_1253.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160711_1333.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160711_1402.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160711_1448.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160711_1519.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160715_.root'] = (4.50, 2.356, -4)

        run_files[run]['nerix_160718_1028.root'] = (4.50, 0.345, -10)

        run_files[run]['nerix_160718_1138.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160718_1207.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160718_1249.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160718_1318.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160718_1350.root'] = (4.50, 2.356, -1)
        
        run_files[run]['nerix_160718_1421.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160718_1452.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160718_1521.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160718_1600.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160718_1651.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160725_1039.root'] = (4.50, 0.345, -10)
        
        run_files[run]['nerix_160725_1149.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160725_1218.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160725_1245.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160725_1313.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160725_1341.root'] = (4.50, 2.356, -1)
        
        run_files[run]['nerix_160725_1411.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160725_1442.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160725_1510.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160725_1540.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160725_1610.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160726_1434.root'] = (4.50, 0.345, -10)
        
        run_files[run]['nerix_160728_1046.root'] = (4.50, 0.345, -10)

        run_files[run]['nerix_160801_1036.root'] = (4.50, 0.345, -10)

        run_files[run]['nerix_160808_1141.root'] = (4.50, 0.345, -10)

        run_files[run]['nerix_160801_1145.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160801_1300.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160801_1328.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160801_1356.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160801_1424.root'] = (4.50, 2.356, -1)
        
        run_files[run]['nerix_160801_1453.root'] = (4.50, 0.345, -5)
        run_files[run]['nerix_160801_1530.root'] = (4.50, 0.700, -5)
        run_files[run]['nerix_160801_1601.root'] = (4.50, 1.054, -5)
        run_files[run]['nerix_160801_1631.root'] = (4.50, 1.500, -5)
        run_files[run]['nerix_160801_1702.root'] = (4.50, 2.356, -5)

        run_files[run]['nerix_160802_0944.root'] = (4.50, 0.345, -2)
        run_files[run]['nerix_160802_1015.root'] = (4.50, 0.700, -2)
        run_files[run]['nerix_160802_1048.root'] = (4.50, 1.054, -2)
        run_files[run]['nerix_160802_1135.root'] = (4.50, 1.500, -2)
        run_files[run]['nerix_160802_1206.root'] = (4.50, 2.356, -2)

        run_files[run]['nerix_160803_1350.root'] = (4.50, 0.345, -2)

        #V5=470V
        run_files[run]['nerix_160803_1453.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160803_1715.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160803_1813.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160804_0659.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160804_0722.root'] = (4.50, 2.356, -1)

        #V5=485V
        run_files[run]['nerix_160804_0949.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160804_1035.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160804_1125.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160804_1205.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160804_1248.root'] = (4.50, 2.356, -1)

        #V5=500V
        run_files[run]['nerix_160804_1355.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160804_1419.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160804_1440.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160804_1457.root'] = (4.50, 1.500, -1)

        #V5=515V
        run_files[run]['nerix_160808_0625.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160808_0701.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160808_0712.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160808_0724.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160808_0736.root'] = (4.50, 2.356, -1)

        #V5=530V
        run_files[run]['nerix_160808_0921.root'] = (4.50, 0.345, -1)
        run_files[run]['nerix_160808_0932.root'] = (4.50, 0.700, -1)
        run_files[run]['nerix_160808_0943.root'] = (4.50, 1.054, -1)
        run_files[run]['nerix_160808_0953.root'] = (4.50, 1.500, -1)
        run_files[run]['nerix_160808_1004.root'] = (4.50, 2.356, -1)



        # Na22 trigger efficiency



        run_files[run]['nerix_160708_1521.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160709_1859.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160710_1735.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160712_1512.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160713_1705.root'] = (4.50, 0.345, -50)

        run_files[run]['nerix_160714_1127.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160715_1527.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160716_1245.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160717_1438.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160722_1759.root'] = (4.50, 0.345, -50)

        run_files[run]['nerix_160726_1635.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160728_1344.root'] = (4.50, 0.345, -50)
        run_files[run]['nerix_160802_1530.root'] = (4.50, 0.345, -50)
        #run_files[run]['nerix_160803_0913.root'] = (4.50, 0.345, -50)




        # 0.19"
        """
        run_files[run]['nerix_160323_1653.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160323_1657.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160324_1448.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160324_1454.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        """


        # 0.25"
        # 500 V/cm

        run_files[run]['nerix_160330_1726.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160330_1747.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160331_1629.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160401_1628.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160403_0936.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})

        run_files[run]['nerix_160405_1145.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160405_1604.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160406_0947.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160406_1556.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160407_0916.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})

        run_files[run]['nerix_160407_1629.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160407_1712.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        
        # MUST CHECK BELOW 12/12/16
        #run_files[run]['nerix_160408_0916.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        #run_files[run]['nerix_160409_1509.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        #run_files[run]['nerix_160412_1153.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})

        #run_files[run]['nerix_160412_1526.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})
        #run_files[run]['nerix_160413_0913.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]}) # raw data deleted
        #run_files[run]['nerix_160413_1500.root'] = (4.50, 1.054, {2300:[0,1], 3000:[2,3]})





        # 200 V/cm

        run_files[run]['nerix_160419_1032.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160419_1535.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]}) # possibly bad (high rate)...
        run_files[run]['nerix_160420_0913.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160421_0917.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        #run_files[run]['nerix_160421_1744.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]}) # files need correcting


        run_files[run]['nerix_160503_1134.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160503_1646.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]}) # high rate
        run_files[run]['nerix_160504_0915.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]}) # high rate
        run_files[run]['nerix_160504_1544.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160504_1637.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})

        run_files[run]['nerix_160504_2002.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160505_1759.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160506_0937.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160506_1632.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160507_2007.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})
        
        run_files[run]['nerix_160508_1600.root'] = (4.50, 0.345, {2300:[0,1], 3000:[2,3]})



        # 1000 V/cm
        #run_files[run]['nerix_160422_1439.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]}) # some corrupted files
        #run_files[run]['nerix_160422_1632.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]}) # some corrupted files
        run_files[run]['nerix_160425_1729.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160426_1746.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160426_1843.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})

        run_files[run]['nerix_160427_1729.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160428_1318.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160429_1500.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160429_1651.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})
        run_files[run]['nerix_160430_1209.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})

        run_files[run]['nerix_160501_1049.root'] = (4.50, 2.356, {2300:[0,1], 3000:[2,3]})



        # 7 and 10 keV
        # 200 V/cm
        run_files[run]['nerix_160510_1754.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160511_0914.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160511_1620.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160511_1650.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160512_1718.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})

        run_files[run]['nerix_160513_0712.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160513_1557.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160515_0818.root'] = (4.50, 0.345, {3500:[0,1], 4500:[2,3]})


        # 7 and 10 keV
        # 500 V/cm

        run_files[run]['nerix_160517_1557.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160518_0702.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160519_1344.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160519_1557.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]}) # need to process
        run_files[run]['nerix_160520_1827.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]})

        run_files[run]['nerix_160520_1851.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160521_1321.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160522_1257.root'] = (4.50, 1.054, {3500:[0,1], 4500:[2,3]})



        # 7 and 10 keV
        # 1000 V/cm

        run_files[run]['nerix_160524_1551.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160524_1715.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160525_1548.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160526_0954.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160601_1341.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})

        run_files[run]['nerix_160602_1237.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160604_1409.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160607_1359.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160608_0934.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160608_1702.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})

        run_files[run]['nerix_160609_1526.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})
        run_files[run]['nerix_160609_1724.root'] = (4.50, 2.356, {3500:[0,1], 4500:[2,3]})


        # 15 and 20 keV
        # 200 V/cm

        run_files[run]['nerix_160613_1737.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160613_1800.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160615_1444.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160615_1638.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160616_0913.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})

        run_files[run]['nerix_160617_1145.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160618_1859.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160619_1445.root'] = (4.50, 0.345, {5300:[0,1], 6200:[2,3]})



        # 15 and 20 keV
        # 500 V/cm

        run_files[run]['nerix_160621_1417.root'] = (4.50, 1.054, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160622_0908.root'] = (4.50, 1.054, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160622_1239.root'] = (4.50, 1.054, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160623_1326.root'] = (4.50, 1.054, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160623_1656.root'] = (4.50, 1.054, {5300:[0,1], 6200:[2,3]})

        run_files[run]['nerix_160624_1300.root'] = (4.50, 1.054, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160626_1433.root'] = (4.50, 1.054, {5300:[0,1], 6200:[2,3]})


        # 15 and 20 keV
        # 1000 V/cm

        run_files[run]['nerix_160628_1349.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160628_1646.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160629_0811.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160630_1142.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160701_0928.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})

        run_files[run]['nerix_160702_1425.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160703_1133.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160704_0908.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
        run_files[run]['nerix_160704_1532.root'] = (4.50, 2.356, {5300:[0,1], 6200:[2,3]})
