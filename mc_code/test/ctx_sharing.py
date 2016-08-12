import pycuda.driver as drv
import os, time
import pycuda

drv.init()
#import pycuda.autoinit
dev = drv.Device(0)

global context
context = dev.make_context(drv.ctx_flags.SCHED_AUTO)

local_ctx = context

print 'hello'

pid = os.fork()

if pid == 0:
	print globals()
	thread_ctx = local_ctx.push()
	print thread_ctx
else:
	time.sleep(5)

