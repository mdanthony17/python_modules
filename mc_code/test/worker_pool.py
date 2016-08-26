from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
import numpy as np
from production_code import cuda_full_observables_production_code

#drv.init()

import threading, time, os, subprocess, sys
#from multiprocessing import Pool, Queue, Process
from Queue import Queue

num_workers = 1

class gpu_pool:
	def __init__(self, num_gpus, grid_dim, block_dim):
		self.num_gpus = num_gpus
		self.grid_dim = grid_dim
		self.block_dim = block_dim
		self.observables_code = cuda_full_observables_production_code
	
		
		self.alive = True
		self.q_gpu = Queue()
		for i in xrange(self.num_gpus):
			self.q_gpu.put(i)
		
		self.q_in = Queue()
		self.q_out = Queue()
		self.l_dispatcher_threads = []
		self.dispatcher_dead_time = 0.5
		for i in xrange(self.num_gpus):
			if self.q_gpu.empty():
				break
			print 'starting worker'
			self.l_dispatcher_threads.append(threading.Thread(target=self.dispatcher, args=[self.q_gpu.get()]))
			self.l_dispatcher_threads[-1].start()



	def dispatcher(self, device_num):
		try:
			drv.init()
			print device_num
			dev = drv.Device(device_num)
		except:
			sys.exit()
		ctx = dev.make_context()
		print dev.name()
		
		
		# source code
		self.l_gpu_modules[device_num] = pycuda.compiler.SourceModule(cuda_full_observables_production_code, no_extern_c=True)
		self.l_gpu_funcs[device_num] = self.l_gpu_modules[device_num].get_function('gpu_full_observables_production')
		
		grid_dim = self.grid_dim
		block_dim = self.block_dim
		num_entries = grid_dim*block_dim
		
		aEnergy = np.full(num_entries, 10., dtype=np.float32)
		aEnergy_gpu = pycuda.gpuarray.to_gpu_async(aEnergy)
		#a_gpu_energy = drv.to_device(aEnergy)
		
		
		#print '\n\ndebug'
		#print self.l_gpu_funcs
		#print ctx
		#print dev
		#print self.l_gpu_streams
		#print '\n\n'
		
		# wrap up function
		# modeled off of pycuda's autoinit
		def _finish_up(ctx):
			print 'wrapping up'
			ctx.pop()
		
			from pycuda.tools import clear_context_caches
			clear_context_caches()
		
		import atexit
		#atexit.register(_finish_up, [ctx])
		atexit.register(ctx.pop)
		
	
		while self.alive:
			if not self.q_in.empty():
				task, args, id_num = self.q_in.get()
				
				#self.lock.acquire()
				aS1 = np.full(num_entries, -2, dtype=np.float32)
				#self.a_gpu_s1 = drv.to_device(self.aS1)
				aS1_gpu = pycuda.gpuarray.to_gpu_async(aS1)
				aS2 = np.full(num_entries, -2, dtype=np.float32)
				#self.a_gpu_s2 = drv.to_device(self.aS2)
				aS2_gpu = pycuda.gpuarray.to_gpu_async(aS2)
				
				#self.lock.release()
				#print args
				for i in xrange(len(args)):
					#print args[i]
					args[i] = drv.In(args[i])
				
				#args = args[0:4] + [a_gpu_energy] + args[4:]
				#args = args[0:2] + [drv.InOut(self.aS1), drv.InOut(self.aS2), a_gpu_energy] + args[4:]
				args = args[0:2] + [aS1_gpu.gpudata, aS2_gpu.gpudata, aEnergy_gpu.gpudata] + args[2:]
			
				#print args
			
				r_value = self.l_gpu_funcs[device_num](*args, grid=(grid_dim,1), block=(block_dim,1,1))
				if task == sys.exit:
					apply(task, [0])
			
				aS1_gpu.get(ary=aS1)
				aS2_gpu.get(ary=aS2)
				
				
				#print id_num, dev.name(), aS2
				#sys.stdout.flush()
				
				self.q_out.put((id_num, r_value))
			else:
				time.sleep(self.dispatcher_dead_time)
	
	
		# add at_exit function like it autoinit



	def map(self, func, l_args):
		start_time = time.time()
		
		print l_args
		
		for id_num, args in enumerate(l_args):
			self.q_in.put((func, args, id_num))

		while not self.q_in.empty():
			time.sleep(0.1)
		print 'Time calling function: %.3e' % (time.time() - start_time)
		sys.stdout.flush()
		
		l_q = list(self.q_out.queue)
		print l_q


	def close(self):
		self.map(sys.exit, [[] for i in xrange(self.num_gpus)])
		print 'Closed children'
		time.sleep(5)


def print_num(i):
	print i
	return i


grid_dim, block_dim = 8192/2, 512
num_entries = grid_dim*block_dim

#aS1 = np.full(num_entries, -1, dtype=np.float32)
#aS2 = np.full(num_entries, -1, dtype=np.float32)

seed = np.asarray(int(time.time()*1000), dtype=np.int32)
num_trials = np.asarray(num_entries, dtype=np.int32)
photonYield = np.asarray(5, dtype=np.float32)
chargeYield = np.asarray(5, dtype=np.float32)
excitonToIonRatio = np.asarray(.5, dtype=np.float32)
g1Value = np.asarray(.12, dtype=np.float32)
extractionEfficiency = np.asarray(.9, dtype=np.float32)
gasGainValue = np.asarray(30., dtype=np.float32)
gasGainWidth = np.asarray(5., dtype=np.float32)
speRes = np.asarray(.7, dtype=np.float32)
intrinsicResS1 = np.asarray(.3, dtype=np.float32)
intrinsicResS2 = np.asarray(.1, dtype=np.float32)


#tArgs = [drv.In(seed), drv.In(num_trials), drv.In(photonYield), drv.In(chargeYield), drv.In(excitonToIonRatio), drv.In(g1Value), drv.In(extractionEfficiency), drv.In(gasGainValue), drv.In(gasGainWidth), drv.In(speRes), drv.In(intrinsicResS1), drv.In(intrinsicResS2)]

tArgs = [seed, num_trials, photonYield, chargeYield, excitonToIonRatio, g1Value, extractionEfficiency, gasGainValue, gasGainWidth, speRes, intrinsicResS1, intrinsicResS2]


g_pool = gpu_pool(num_workers, grid_dim, block_dim)
g_pool.map(print_num, [list(tArgs) for i in xrange(100)])

print 'finished map'

g_pool.close()


