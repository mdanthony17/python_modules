from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
import numpy as np

#drv.init()

import threading, time, os, subprocess
from multiprocessing import Pool, Queue, Process


num_workers = 2

def night_night(i):
	print 'night night %d'

class pool_forks:
	def __init__(self):
		pass

	def start_workers(self):
		self.l_processes = []
		self.relation = None
		for i in xrange(num_workers):
			pid = os.fork()
			if pid == 0:
				self.relation = 'c'
				break
			self.relation = 'p'
			self.l_processes.append(pid)

		if pid != 0:
			print 'parent sleeping'
			print 'parent id: %d' % os.getpid()
			time.sleep(5)
			print 'parent awake!\n\n'


	def check_workers_alive(self):
		if self.relation == 'p':
			for pid in self.l_processes:
				print 'checking %d' % pid
				try:
					os.kill(pid, 0)
					print 'process was alive'
				except OSError:
					print 'process DEAD'
	
	
	
	def set_queue(self):
		if self.relation == 'p':
			self.q = Queue()
			for i in xrange(5):
				self.q.put(i)


	def call_function_worker(self, func):
		# use queue with a while condition
		
		current_q = self.q
		
		
		while not self.q.empty():
			if self.relation == 'c':
				func(self.q.get(i))



pool = pool_forks()
pool.start_workers()
pool.set_queue()
pool.call_function_worker(night_night)
pool.check_workers_alive()
