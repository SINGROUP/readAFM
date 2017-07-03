First 4 bits is an integer which specifies the number of orientations present in the file (default 100)
Next BLOCK contains information about one particular orientation
This BLOCK is as follows:
	First 4 bits specifies the number of atoms in this orientation. Usually same for all the orientations(because they're all of the same molecule)
		1 bit which is a character which is the atomic symbol
		4 bits which are a FLOAT which gives the X coordinate of the above mentioned atom
		4 bits which are a FLOAT which gives the Y coordinate of the above mentioned atom
		4 bits which are a FLOAT which gives the Z coordinate of the above mentioned atom
	The above block repeats for each atom in the molecule. That means 13 bits per atom
The next block specifies the information about the grid
4 bits FLOAT widthZ
4 bits FLOAT widthX
4 bits FLOAT widthY
4 bits FLOAT dz	
4 bits FLOAT dx
4 bits FLOAT dy
4 bits INT divZ: number of divisions
4 bits INT divX: number of divisions
4 bits INT divY: number of divisions
The next block is composed of all the fZ values
This is sorted in the order Z > X > Y
For example, it will be as follows(each 4 bits are a float):
	Z X Y
	0 0 0
	0 0 1
	0 1 0
	0 1 1
	1 0 0
	1 0 1
	1 1 0
	1 1 1
The whole of the above will be repeated for each orientation. This number is the first INT in the file
