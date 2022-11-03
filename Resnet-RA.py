### INPUTS

epochs = 40

shuffle_squence_order = False

train_set = epochs[:30]

test_set = epochs[30:]


#----------------------------
currentTimeslot = 0
phaseAtInitialTimeslot = 23
ModulationProbabilityFactor = [0 0 0 0 0 0 0 0 0 0 0 0.2 0.4 0.8 1.6 3.2 6.4 12.8 25.6 12.8 6.4 3.2 1.6 0.8 0.4 0.2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
IntegerCollectionForInputStateGeneration = np.arange(10000)

def defineObjectCategories():
	availableProbabilities = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 17 17 17 17 17 18 18 18 18 18 19 19 19 19 19 20 20 20 20 20 21 21 21 21 21 22 22 22 22 22 23 23 23 23 23 24 24 24 24 24 25 25 25 25 25 26 26 26 26 26 27 27 27 27 27 28 28 28 28 28 29 29 29 29 29 30 30 30 30 30 31 31 31 31 31 32 32 32 32 32 33 33 33 33 33 34 34 34 34 34 35 35 35 35 35 36 36 36 36 36 37 37 37 37 37 38 38 38 38 38 39 39 39 39 39 40 40 40 40 40 41 41 41 41 42 42 42 42 43 43 43 43 44 44 44 44 45 45 45 45 46 46 46 46 47 47 47 47 48 48 48 48 49 49 49 49 50 50 50 50 51 51 51 51 52 52 52 52 53 53 53 53 54 54 54 54 55 55 55 55 56 56 56 56 57 57 57 57 58 58 58 58 59 59 59 59 60 60 60 60 61 61 61 61 62 62 62 62 63 63 63 63 64 64 64 64 65 65 65 65 66 66 66 66 67 67 67 67 68 68 68 68 69 69 69 69 70 70 70 70 71 71 71 71 72 72 72 72 73 73 73 73 74 74 74 74 75 75 75 75 76 76 76 76 77 77 77 77 78 78 78 78 79 79 79 79 80 80 80 80 81 81 81 82 82 82 83 83 83 84 84 84 85 85 85 86 86 86 87 87 87 88 88 88 89 89 89 90 90 90 91 91 91 92 92 92 93 93 93 94 94 94 95 95 95 96 96 96 97 97 97 98 98 98 99 99 99 100 100 100 101 101 101 102 102 102 103 103 103 104 104 104 105 105 105 106 106 106 107 107 107 108 108 108 109 109 109 110 110 110 111 111 111 112 112 112 113 113 113 114 114 114 115 115 115 116 116 116 117 117 117 118 118 118 119 119 119 120 120 120 121 121 122 122 123 123 124 124 125 125 126 126 127 127 128 128 129 129 130 130 131 131 132 132 133 133 134 134 135 135 136 136 137 137 138 138 139 139 140 140 141 141 142 142 143 143 144 144 145 145 146 146 147 147 148 148 149 149 150 150 151 151 152 152 153 153 154 154 155 155 156 156 157 157 158 158 159 159 160 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200]
	#each row is a new category
	CategorySpikeProbabilities = np.empty((30,400))
	for j in range(30):
		for i in range(400):
			CategorySpikeProbabilities[j,i] = np.random.choice(availableProbabilities, 1)
	return CategorySpikeProbabilities


def getSpikesInNextTimeslot(category, secondCategory, thirdCategory):
	category_size = category.shape()
	currentTimeslot += 1
	currentPhase = phaseAtInitialTimeslot + (currentTimeslot -1)
	currentPhase -= 75*(currentPhase//75)
	if currentPhase == 0:
		currentPhase = 75
	secondPhase = currentPhase + 25
	if secondPhase > 75:
		secondPhase -= 75
	thirdPhase = secondPhase + 25
	if thirdPhase > 75:
		thirdphase -= 75
	inputArray = np.empty()
	for i in category_size:
		currentInputSpikeProbability =  (category[i] * ModulationProbabilityFactor[currentPhase]) + (secondCategory[i] * ModulationProbabilityFactor[secondPhase]) + (thirdCategory[i] * ModulationProbabilityFactor[thirdphase])
		if currentInputSpikeProbability > np.random.choice(IntegerCollectionForInputStateGeneration,1):
			inputArray.append(1)
		else:
			inputArray.append(0)
	return inputArray 

# inputSourceCategories = np.empty((30,400))
# for i in range(30):
# 	inputSourceCategories[i,:] = getSpikesInNextTimeslot(CategorySpikeProbabilities[i,:])


def presentTripleCategoryInstance(firstCategoryInputSource, secondCategoryInputSource, thirdCategoryInputSource):
	for i in range(600):
		inPuts =  getSpikesInNextTimeslot(firstCategoryInputSource,secondCategoryInputSource,thirdCategoryInputSource)
	return inPuts #[600X400]


## SEND INPUTS TO BRAIN --- CHECK ON THIS
for epoch in epochs:
	outs_epoch = []
	for i in np.arange(0,30,3):
		ins = presentTripleCategoryInstance(inputSourceCategories[i,:],inputSourceCategories[i+1,:],inputSourceCategories[i+2,:])
		for j in range(600):
			outs_epoch.append(brain(ins[j,:]))


### OUTPUTS
output_nodes = 30 #MORE OUTPUT NODES PER CATEGORIES VIA COMBO --guess 3 cats at a time 

first = np.argmax(output_layer)

temp_out = np.delete(output_layer.copy(), first)

second = np.argmax(temp_out, first)

third = np.argmax(np.delete(temp_out, second))


#RESNET-----------------------










