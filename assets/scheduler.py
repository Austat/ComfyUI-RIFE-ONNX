import queue
from .contracts import WorkItem
class Scheduler:
 @staticmethod
 def build(pair_count,batch_size):
  q=queue.Queue()
  for start in range(0,pair_count,batch_size):q.put(WorkItem(start,min(start+batch_size,pair_count)))
  return q
