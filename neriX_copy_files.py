import sys, os
import neriX_analysis, neriX_datasets, neriX_config
from subprocess import call
import click


def copy_all_files(run, anodeSetting, cathodeSetting, degreeSetting):
	lFilesToDownload = neriX_analysis.pull_all_files_given_parameters(run, anodeSetting, cathodeSetting, degreeSetting)
	
	localPathForData = neriX_config.pathToData + '/run_%d/' % run

	print '\n\nCode will produce a large amount of output as it checks multiple directories for each file so please be certain that files were downloaded without issue.\n\n'

	print '\n\n'
	print lFilesToDownload
	print '\n\n'

	response = raw_input('\n\nWill download the above files.  Enter "y" to continue: ')
	if response != 'y':
		return

	# download files in rsync
	for file in lFilesToDownload:
		for path in neriX_config.lPathsToDataArchive:
			call(['rsync', '-avz', 'mdanthony@128.59.171.68:%s/%s' % (path, file), localPathForData])

	lFilesInData = os.listdir(localPathForData)
	bAllOkay = True
	for file in lFilesToDownload:
		if not (file in lFilesInData):
			neriX_analysis.failure_message('%s not found locally!\n' % file)
			bAllOkay = False

	if bAllOkay:
		neriX_analysis.success_message('All files found locally!')



def copy_single_file(filename):
	if filename[-5:] != '.root':
		filename += '.root'
	
	localPathForData = neriX_config.pathToData + '/run_%d/' % run
	
	lFilesInData = os.listdir(localPathForData)
	if filename in lFilesInData:
		neriX_analysis.success_message('%s found - will not download')
		return

	print '\n\nCode will produce a large amount of output as it checks multiple directories for each file so please be certain that files were downloaded without issue.\n\n'

	response = raw_input('\n\nWill download %s.  Enter "y" to continue: ' % filename)
	if response != 'y':
		return

	for path in neriX_config.lPathsToDataArchive:
		call(['rsync', '-avz', 'mdanthony@128.59.171.68:%s/%s' % (path, filename), localPathForData])

	if filename in lFilesInData:
		neriX_analysis.success_message('%s found locally!' % filename)
	else:
		neriX_analysis.failure_message('%s not found locally!\n' % file)



if __name__ == '__main__':
	if len(sys.argv) != 5:
		print 'Use is python neriX_copy_files.py <run> <anode setting> <cathode setting> <degree setting>'
		sys.exit()

	run = int(sys.argv[1])
	anodeSetting = float(sys.argv[2])
	cathodeSetting = float(sys.argv[3])
	degreeSetting = float(sys.argv[4])

	copy_all_files(run, anodeSetting, cathodeSetting, degreeSetting)
