import numpy as np
grid = None
def checkRow(i,k):
	global grid
	for j in range(9):
		if grid[i][j] == k:
			return False
	return True

def checkColumn(i,k):
	global grid
	for j in range(9):
		if grid[j][i] == k:
			return False
	return True

def checkBox(i,j,k):
	global grid
	i = i - i%3
	j = j - j%3
	for a in range(i,i+3):
		for b in range(j,j+3):
			if grid[a][b] == k:
				return False
	return True

def solve(pos):
	global grid
	if pos==81:
		return True
	i=pos//9
	j=pos%9
	if grid[i][j]!=0 :
		return solve(pos+1);
	for k in range(1,10):
		if checkRow(i,k) and checkColumn(j,k) and checkBox(i, j, k):
			grid[i][j]=k
			if solve(pos+1):
				return True	
	
	grid[i][j] = 0
	return False


def backtracingSolver(sudoku):
	global grid
	grid = sudoku
	solve(0)
	return grid
