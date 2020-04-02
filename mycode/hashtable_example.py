import random
import time

class LinearMap(object):

	def __init__(self):
		self.items = []

	def add(self, k, v):
		self.items.append( (k, v))

	def get(self, k):
		for key, val in self.items:
			if key == k:
				return val

		raise KeyError

class BetterMap(object):

	def __init__(self, n=100):
		self.maps = []
		for i in range(n):
			self.maps.append(LinearMap())

	def find_map(self, k):
		index = hash(k) % len(self.maps)
		return self.maps[index]

	def add(self, k, v):
		m = self.find_map(k)
		m.add(k,v)

	def get(self, k):
		m = self.find_map(k)
		return m.get(k)

class HashMap(object):

	def __init__(self):
		self.maps = BetterMap(2)
		self.num = 0

	def get(self, k):
		return self.maps.get(k)

	def add(self, k, v):
		if self.num == len(self.maps.maps):
			self.resize()

		self.maps.add(k, v)
		self.num += 1

	def resize(self):
		new_maps = BetterMap(self.num * 2)

		for m in self.maps.maps:
			for k, v in m.items:
				new_maps.add(k, v)

		self.maps = new_maps


def make_map(M, n=10000):

	t = []

	for i in range(n):

		x = random.randint(1,n)
		M.add(i, x)

	for j in range(n):

		y = random.randint(1,n)
		try:
			t.append(M.get(y))
		except:
			pass

	return t


if __name__ == '__main__':
	L = LinearMap()
	B = BetterMap()
	H = HashMap()

	start = time.process_time()
	h_l = make_map(H)
	end = time.process_time()
	print('h_l: ', len(h_l), ' ints')
	print('hash map: ', end-start, ' s')

	start = time.process_time()
	t_b = make_map(B)
	end = time.process_time()
	print('t_b: ', len(t_b), ' ints')
	print('better map: ', end-start, ' s')

	start = time.process_time()
	t_l = make_map(L)
	end = time.process_time()
	print('t_l: ', len(t_l), ' ints')
	print('linear map: ', end-start, ' s')
